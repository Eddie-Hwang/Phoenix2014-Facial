from model.helpers import *
from dataset.data import *
from torch.utils.tensorboard import SummaryWriter
from model.loss import MSELoss, CrossEntropyLoss
from model.builders import build_optimizer, build_scheduler, build_gradient_clipper
from dataset.batch import Batch
from tqdm import tqdm

import time
import queue

class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model, config):
        """
        Creates a new TrainManager for a model, specified as in configuration.
        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config['training']

        # Model related
        self.model = model
        self.txt_pad_index = self.model.txt_pad_token
        self.bos_pad_index = self.model.txt_bos_token
        self.gls_pad_index = self.model.gls_pad_token

        self.do_translation = train_config.get('translation_loss_weight', 1.0) > 0.0
        self.do_generation = train_config.get('generation_loss_weight', 1.0) > 0.0

        # Loss related
        self.translation_loss_function = CrossEntropyLoss(
            pad_idx=self.model.gls_pad_token,
        )
        self.translation_loss_weight = train_config['translation_loss_weight']
        self.generation_loss_function = MSELoss()
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
        self.logging_display = train_config.get('logging_display', False)
        self.tb_writer = SummaryWriter(log_dir=train_config['model_dir'] + "/tensorboard/")

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
            "ppl",
            "translation_loss",
            "recognition_loss",
        ]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf", "rouge"]:
                assert self.do_translation
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
        # self.total_txt_tokens = 0
        # self.total_gls_tokens = 0
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
        total_loss = translation_loss + generation_loss
        
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
    
    def train_and_validation(self, train_data, valid_data):
        # Train dataset
        train_iter = make_data_iter(
            dataset=train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            train=True,
            shuffle=self.shuffle,
        )
        # Valid dataset
        val_iter = make_data_iter(
            dataset=valid_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            train=False,
        )

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
            
            for batch in tqdm(iter(train_iter), desc='- (Training)', leave=False):
                batch = Batch(
                    torch_batch=batch,
                    txt_pad_index=self.model.txt_pad_token,
                    trg_pad_token=self.model.trg_pad_token,
                    use_cuda=self.use_cuda,
                )

                update = (count == 0)
                tr_translation_loss, tr_generation_loss = self._train_batch(batch, update)

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
                        tr_generation_loss, 
                        self.steps
                    )

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "step"
                    and update
                ):
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
                    if self.do_translation:
                        log_out += "Batch Generation Loss: {:10.6f} => ".format(
                            tr_generation_loss
                        )
                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)

            if self.steps % self.validation_freq == 0 and update:
                # Set model on evaluation mode
                self.model.eval()
                
                # Don't track gradients during validation
                with torch.no_grad():
                    batch_translation_loss = 0
                    batch_generation_loss = 0
                    for batch in tqdm(iter(val_iter), desc='- (Validation)', leave=False):
                        batch = Batch(
                            torch_batch=batch,
                            txt_pad_index=self.model.txt_pad_token,
                            trg_pad_token=self.model.trg_pad_token,
                            use_cuda=self.use_cuda,
                        )

                        # Foward
                        val_translation_loss, val_generation_loss = self.model.train_batch(
                                                                        batch=batch,
                                                                        translation_loss_function=self.translation_loss_function,
                                                                        translation_loss_weight=self.translation_loss_weight,
                                                                        generation_loss_function=self.generation_loss_function,
                                                                        generation_loss_weight=self.generation_loss_weight,
                        )

                        batch_translation_loss += val_translation_loss
                        batch_generation_loss += val_generation_loss

                        # Write loss on tensorboard
                        if self.do_translation:
                            self.tb_writer.add_scalar(
                                'Translation_loss(Valid)', 
                                val_translation_loss, 
                                self.steps
                            )
                        if self.do_generation:
                            self.tb_writer.add_scalar(
                                'generation_loss(Valid)', 
                                val_generation_loss, 
                                self.steps
                            )
                    # Logging
                    log_out = '[Validation] '
                    if self.do_translation:
                        log_out += 'Valid Translation Loss: {:10.6f} => '.format(
                            batch_translation_loss/len(val_iter)
                        )
                    if self.do_translation:
                        log_out += "Valid Generation Loss: {:10.6f} => ".format(
                            batch_generation_loss/len(val_iter)
                        )
                    self.logger.info(log_out)
