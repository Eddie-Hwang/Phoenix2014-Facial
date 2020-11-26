from model.helpers import *
from dataset.data import *
from torch.utils.tensorboard import SummaryWriter
from model.loss import MSELoss, CrossEntropyLoss
from model.builders import build_optimizer, build_scheduler, build_gradient_clipper
from model.prediction import validate_on_data
from model.batch import Batch
from tqdm import tqdm

import time
import queue

class TrainManager:

    def __init__(self, model, config):
        
        train_config = config['training']

        # Model related
        self.model = model
        self.txt_pad_index = self.model.txt_pad_token
        self.bos_pad_index = self.model.txt_bos_token
        self.gls_pad_index = self.model.gls_pad_token

        self.do_translation = train_config.get('translation_loss_weight', 1.0) > 0.0
        self.do_generation = train_config.get('generation_loss_weight', 1.0) > 0.0

        self.use_custom_loss = train_config.get('use_custom_loss', True)

        # Loss related
        self.translation_loss_function = CrossEntropyLoss(
            pad_idx=self.model.gls_pad_token,
        )
        self.translation_loss_weight = train_config['translation_loss_weight']
        self.generation_loss_function = MSELoss(
            use_custom_loss=self.use_custom_loss
        )
        # To evaluate
        self.eval_generation_loss_function = MSELoss(
            use_custom_loss=False
        )
        self.generation_loss_weight = train_config['generation_loss_weight']

        # Model directory and storing related
        self.model_dir = make_model_dir(
            model_dir=train_config['model_dir'],
            overwrite=train_config.get('overwrite', False)
        )
        self.logger = make_logger(
            model_dir=train_config['model_dir'],
        )
        self.logging_freq = train_config.get('logging_freq', 100)
        self.val_logging_freq = train_config.get('')
        self.logging_display = train_config.get('logging_display', False)
        self.tb_writer = SummaryWriter(log_dir=train_config['model_dir'] + "/tensorboard/")

        # Iteration related
        self.epochs = train_config['epochs']
        self.batch_size = train_config['batch_size']
        self.batch_type = train_config.get('batch_type', 'sentence')
        self.shuffle = train_config.get('shuffle', True)
    
        # Optimization related
        self.last_best_lr = train_config.get("learning_rate", -1)
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(
            config=train_config, parameters=model.parameters()
        )
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # Validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 100)
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.ckpt_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf", "wer", "rouge"]:
            raise ValueError(
                "Invalid setting for 'eval_metric': {}".format(self.eval_metric)
            )
        self.early_stopping_metric = train_config.get('early_stopping_metric', 'eval_metric')

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in [
            "translation_loss",
            "generation_loss",
        ]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.do_translation and self.eval_metric in ["bleu", "chrf", "rouge"]:
                self.minimize_metric = False
            else:  # eval metric that has to get minimized (not yet implemented)
                self.minimize_metric = True
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        # Learning rate scheduling related
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        # Training statistics related
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}
        # comparison function for scores
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        # Cuda allocate
        self.use_cuda = train_config['use_cuda']
        if self.use_cuda:
            self.model.cuda()
            if self.do_translation:
                self.translation_loss_function.cuda()
            if self.do_generation:
                self.generation_loss_function.cuda()

    def _train_batch(self, batch, update):
        # We do not need to normalize loss
        # MSE already provide normalized output
        translation_loss, generation_loss = self.model.train_batch(
            batch=batch,
            translation_loss_function=self.translation_loss_function,
            translation_loss_weight=self.translation_loss_weight,
            generation_loss_function=self.generation_loss_function,
            generation_loss_weight=self.generation_loss_weight
        )

        _generation_loss, _, _, _ = generation_loss

        # If we use encoder loss together, then backwards
        # both tranlation and generation loss
        if self.do_translation and self.do_generation:
            total_loss = translation_loss + _generation_loss
        # elif not(self.do_generation):
        #     total_loss = translation_loss
        elif not(self.do_translation):
            total_loss = _generation_loss
        
        # Backward
        total_loss.backward()
        
        # Clipping
        if self.clip_grad_fun is not None:
            self.clip_grad_fun(params=self.model.parameters())
        
        # Optimizing
        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1

        return translation_loss, generation_loss

    def _save_checkpoint(self):
        model_path = '{}/{}.ckpt'.format(self.model_dir, self.steps)
        state = {
            'steps': self.steps,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
            if self.scheduler is not None else None
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )

        self.ckpt_queue.put(model_path)

        # create/modify symbolic link for best checkpoint
        symlink_update(
            "{}.ckpt".format(self.steps), "{}/best.ckpt".format(self.model_dir)
        )
    
    def train_and_validation(self, train_data, valid_data):
        # Train dataset
        train_iter = make_data_iter(
            dataset=train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            train=True,
            shuffle=self.shuffle,
        )

        # Iteration
        for epoch in range(self.epochs):
            self.logger.info('EPOCH {}'.format(epoch + 1))
            # Learning rate scheduler
            if self.scheduler is not None and self.scheduler_step_at == 'epoch':
                self.scheduler.step(epoch=epoch)
            # Set model train mode
            self.model.train()
            # Set start time
            start = time.time()
            # Set count
            count = self.batch_multiplier - 1
            
            # Training
            with tqdm(total=len(train_iter), desc='- (Training)', leave=False) as pbar:
                for batch in iter(train_iter):
                    batch = Batch(
                        torch_batch=batch,
                        txt_pad_index=self.model.txt_pad_token,
                        trg_pad_token=self.model.trg_pad_token,
                        use_cuda=self.use_cuda,
                    )
                    update = (count == 0)
                    tr_translation_loss, tr_generation_loss = self._train_batch(batch, update)

                    _tr_generation_loss, mse_loss, cont_loss, rotation_loss = tr_generation_loss

                    # Write loss on tensorboard
                    if self.do_translation:
                        self.tb_writer.add_scalar(
                            'Translation_loss(Train)', 
                            tr_translation_loss, 
                            self.steps
                        )
                    if self.do_generation:
                        self.tb_writer.add_scalar(
                            'generation_loss(Train)', 
                            _tr_generation_loss, 
                            self.steps
                        )
                        # self.tb_writer.add_scalar(
                        #     'mse_loss(Train)', 
                        #     mse_loss, 
                        #     self.steps
                        # )
                        # self.tb_writer.add_scalar(
                        #     'cont_loss(Train)', 
                        #     cont_loss, 
                        #     self.steps
                        # )
                        # self.tb_writer.add_scalar(
                        #     'rotation_loss(Train)', 
                        #     rotation_loss, 
                        #     self.steps
                        # )

                    count = self.batch_multiplier if update else count
                    count -= 1

                    if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                        self.scheduler.step()

                    # Log learning process
                    if self.steps % self.logging_freq == 0 and update:
                        log_out = "[Epoch: {:03d} Step: {:08d}] ".format(
                            epoch + 1, self.steps,
                        )
                        if self.do_translation:
                            log_out += 'Batch Translation Loss: {:10.6f} => '.format(
                                tr_translation_loss
                            )
                        if self.do_generation:
                            log_out += "Batch Generation Loss: {:10.6f} => ".format(
                                _tr_generation_loss
                            )
                        log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                        self.logger.info(log_out)
                    
                    pbar.update()
            
            # Validation
            if (self.steps % self.validation_freq == 0) and update:
                valid_result = validate_on_data(
                        model=self.model,
                        batch_size=self.batch_size,
                        batch_type=self.batch_type,
                        valid_data=valid_data,
                        use_cuda=self.use_cuda,
                        translation_loss_function=self.translation_loss_function,
                        translation_loss_weight=self.translation_loss_weight,
                        generation_loss_function=self.eval_generation_loss_function,
                        generation_loss_weight=self.generation_loss_weight,
                        do_translation=self.do_translation,
                        do_generation=self.do_generation,
                        greedy_decode=True,
                        beam_search=False,
                )

                # Set early stopping metric
                if self.early_stopping_metric == 'translation_loss':
                    assert self.do_translation
                    ckpt_score = valid_result['valid_translation_loss']
                elif self.early_stopping_metric == 'generation_loss':
                    assert self.do_generation
                    ckpt_score = valid_result['valid_generation_loss']
                elif self.early_stopping_metric in ['bleu']:
                    assert self.do_translation
                    ckpt_score = valid_result['bleu']['bleu-4']

                new_best = False
                if self.is_best(ckpt_score):
                    self.best_ckpt_score = ckpt_score
                    # self.best_all_ckpt_scores = valid_result['valid_scores']
                    self.best_ckpt_iteration = self.steps
                    self.logger.info(
                            "Hooray! New best validation result [%s]!",
                            self.early_stopping_metric,
                        )
                if self.ckpt_queue.maxsize > 0:
                    self.logger.info("Saving new checkpoint.")
                    new_best = True
                    self._save_checkpoint()

                if self.scheduler is not None and self.scheduler_step_at == 'validation':
                    prev_lr = self.scheduler.optimizer.param_groups[0]['lr']
                    self.scheduler.step(ckpt_score)
                    now_lr = self.scheduler.optimizer.param_groups[0]['lr']

                    if prev_lr != now_lr:
                        if self.last_best_lr != prev_lr:
                            self.stop = True

                # Write loss on tensorboard
                if self.do_translation:
                    self.tb_writer.add_scalar(
                        'Translation_loss(Valid)', 
                        valid_result['valid_translation_loss'], 
                        self.steps
                    )
                if self.do_generation:
                    self.tb_writer.add_scalar(
                        'generation_loss(Valid)', 
                        valid_result['valid_generation_loss'], 
                        self.steps
                    )

                # Logging
                self.logger.info(
                    'Validation result at epoch %3d, step %8d\t'
                    'Translation loss: %4.5f\t'
                    'Generation loss: %4.5f\t'
                    'Eval Metric: %s\n\t'
                    'BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t',
                    epoch + 1,
                    self.steps,
                    valid_result['valid_translation_loss']
                    if self.do_translation else -1,
                    valid_result['valid_generation_loss']
                    if self.do_generation else -1,
                    self.eval_metric.upper(),
                    valid_result['bleu']['bleu4']
                    if self.do_translation else -1,
                    valid_result['bleu']['bleu1']
                    if self.do_translation else -1,
                    valid_result['bleu']['bleu2']
                    if self.do_translation else -1,
                    valid_result['bleu']['bleu3']
                    if self.do_translation else -1,
                    valid_result['bleu']['bleu4']
                    if self.do_translation else -1,
                )

            # Early stopping handler
            if self.stop:
                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "validation"
                    and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break
