import torch
import numpy as np

from dataset.data import make_data_iter, load_data
from model.batch import Batch
from tqdm import tqdm
from model.metrics import bleu, chrf, rouge, wer_list
from model.helpers import load_checkpoint
from model.model import build_model
from model.loss import MSELoss, CrossEntropyLoss

        
def validate_on_data(
    model,
    batch_size,
    batch_type,
    use_cuda,
    valid_data,
    translation_loss_function,
    translation_loss_weight,
    generation_loss_function,
    generation_loss_weight,
    do_translation,
    do_generation,
    greedy_decode,
    beam_search,
    level='word',
    inference=True,
):
    # Valid dataset
    val_iter = make_data_iter(
        dataset=valid_data,
        batch_size=batch_size,
        batch_type=batch_type,
        train=False,
    )
    
    # Set model on evaluation mode
    model.eval()
    
    # Don't track gradients during validation
    with torch.no_grad():
        total_translation_loss = 0
        total_generation_loss = 0
        
        all_gls_outputs = list()
        all_skel_outputs = list()
        
        with tqdm(total=len(val_iter), desc='- (Validation)') as pbar:
            for batch in iter(val_iter):
                batch = Batch(
                    torch_batch=batch,
                    txt_pad_index=model.txt_pad_token,
                    trg_pad_token=model.trg_pad_token,
                    use_cuda=use_cuda,
                )

                # Get tranlation and generation loss for valid data
                batch_translation_loss, batch_generation_loss = model.train_batch(
                            batch=batch,
                            translation_loss_function=translation_loss_function,
                            translation_loss_weight=translation_loss_weight,
                            generation_loss_function=generation_loss_function,
                            generation_loss_weight=generation_loss_weight,
                )
                total_loss, mse, cont, rotation = batch_generation_loss
                if do_translation:
                    total_translation_loss += batch_translation_loss
                if do_generation:
                    total_generation_loss += total_loss

                if inference:
                    trg_input = torch.empty_like(batch.trg_seq)
                    if use_cuda:
                        trg_input = trg_input.cuda()
                else:
                    trg_input = None
                
                batch_gls_prediction, batch_skeleton_output = model.predict_batch(
                        batch=batch,
                        greedy_decode=greedy_decode,
                        beam_search=beam_search,
                        trg_input=trg_input,
                )
                if do_translation:
                    all_gls_outputs.extend(batch_gls_prediction.detach().cpu().numpy())
                if do_generation:
                    all_skel_outputs.extend(batch_skeleton_output.detach().cpu().numpy())
                
                pbar.update()

        # Get text ref
        txt_ref = [' '.join(t) for t in valid_data.text]

        # Get Video name
        vid_name = [t for t in valid_data.id]
        
        if do_translation:
            decoded_glosses = model.gls_vocab.arrays_to_sentences(
                arrays=all_gls_outputs,
                cut_at_eos=True,
            )
            join_char = ' '
            gls_hyp = [join_char.join(t) for t in decoded_glosses]
            gls_ref = [join_char.join(t) for t in valid_data.gls]
            assert len(gls_ref) == len(gls_hyp)
            # Get bleu sroces
            gls_bleu = bleu(
                references=gls_ref,
                hypotheses=gls_hyp,
            )
        
        if do_generation:          
            skel_hyp = all_skel_outputs
            skel_ref = [np.array(skel) for skel in valid_data.landmark]
        
        # Save the results
        results = {}
        
        results['vid_name'] = vid_name
        results['txt_ref'] = txt_ref

        if do_translation:
            results['bleu'] = gls_bleu
            results['valid_translation_loss'] = total_translation_loss
            results['gls_hyp'] = gls_hyp
            results['gls_ref'] = gls_ref
        if do_generation:
            results['valid_generation_loss'] = total_generation_loss
            results['skel_hyp'] = skel_hyp
            results['skel_ref'] = skel_ref

        return results

def test_on_data(
    config,
    ckpt_path,
    tst_data,
    txt_vocab,
    gls_vocab,
):
    # Load configure
    do_generation = config["training"].get("generation_loss_weight", 1.0) > 0.0
    do_translation = config["training"].get("translation_loss_weight", 1.0) > 0.0
    generation_loss_weight = config["training"]['generation_loss_weight']
    translation_loss_weight = config["training"]['translation_loss_weight']
    
    batch_size = config['training']['batch_size']
    batch_type = config['training'].get('batch_type', 'sentence')
    use_cuda = config["training"].get("use_cuda", False)

    # Load trained model checkpoint
    model_ckpt = load_checkpoint(ckpt_path, use_cuda)
    model = build_model(
        config=config['model'],
        txt_vocab=txt_vocab,
        gls_vocab=gls_vocab,
        do_translation=do_translation,
        do_generation=do_generation,
    )
    model.load_state_dict(model_ckpt['model_state'])
    print('[INFO] Trained model is loaded.')

    # Make model cuda
    if use_cuda:
        model.cuda()

    # Define loss fuctions
    if do_translation:
        translation_loss_function = CrossEntropyLoss(
            pad_idx=self.model.gls_pad_token,
        )
        if use_cuda:
            translation_loss_function.cuda()
    else:
        translation_loss_function = None
    if do_generation:
        generation_loss_function = MSELoss(
            use_custom_loss=False
        )
        if use_cuda:
            generation_loss_function.cuda()
    else:
        generation_loss_function = None

    # Testing
    if do_translation:
        pass
    if do_generation:
        # tst_generation_result = {}
        tst_result = validate_on_data(
            model=model,
            batch_size=batch_size,
            batch_type=batch_type,
            use_cuda=use_cuda,
            valid_data=tst_data,
            translation_loss_function=translation_loss_function,
            translation_loss_weight=translation_loss_weight,
            generation_loss_function=generation_loss_function,
            generation_loss_weight=generation_loss_weight,
            do_translation=do_translation,
            do_generation=do_generation,
            greedy_decode=True,
            beam_search=False,
            level='word',
            inference=True,
        )

    return tst_result
