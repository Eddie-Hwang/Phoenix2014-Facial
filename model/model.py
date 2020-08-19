import torch
import torch.nn as nn

from model.encoders import TransformerEncoder
from model.decoders import TransformerDecoder
from model.embeddings import Embeddings
from model.init import *
from opts import *


class SignGenModel(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        txt_embed,
        trg_embed,
        txt_vocab,
        gls_vocab,
        gloss_output_layer,
        do_translation=True,
        do_generation=True,
    ):
        '''
        Create encoder and decoder model

        Args:
            encoder:
            gloss_output_layer:
            decoder:

        '''
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.txt_embed = txt_embed
        self.trg_embed = trg_embed

        self.txt_vocab = txt_vocab
        self.gls_vocab = gls_vocab

        self.txt_bos_token = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_eos_token = self.txt_vocab.stoi[EOS_TOKEN]
        self.txt_pad_token = self.txt_vocab.stoi[PAD_TOKEN]
        
        self.gls_pad_token = self.gls_vocab.stoi[PAD_TOKEN]
        self.trg_pad_token = torch.zeros((PAD_FEATURE_SIZE, ))

        self.gloss_output_layer = gloss_output_layer

        self.do_translation = do_translation
        self.do_generation = do_generation
        
    def forward(
        self,
        txt_input,
        txt_mask=None,
        txt_length=None,
        trg_input=None,
        trg_mask=None,
        gls_input=None,
    ):
        # Encode
        # B x S x Dim
        enc_out, enc_hid = self.encoder(
            embed_src=self.txt_embed(x=txt_input, mask=txt_mask),
            src_length=txt_length,
            mask=txt_mask,
        )

        if self.do_translation:
            gloss_score = self.gloss_output_layer(enc_out) # B x S x gls_dim
            # gloss_prob = gloss_scores.log_softmax(dim=-1) # B x S x gls_dim
            # Get gloss probability upto required sequence length
            gloss_prob = gloss_score[:, :gls_input.size(1), :]
            gloss_prob = gloss_prob.permute(0,2,1) # B x gls_dim x S
        else:
            gloss_prob = None
        
        if self.do_generation:
            dec_outputs = self.decoder(
                    encoder_output=enc_out,
                    encoder_hidden=enc_hid,
                    src_mask=txt_mask,
                    trg_mask=trg_mask,
                    trg_embed=self.trg_embed(trg_input),
            )
        else:
            dec_outputs = None

        return dec_outputs, gloss_prob
        

    def train_batch(
        self,
        batch,
        translation_loss_function,
        translation_loss_weight,
        generation_loss_function,
        generation_loss_weight,
        ):

        # Source data related (src input)
        # txt_input = batch.txt[0] # B x S
        # txt_length = batch.txt[1]
        # txt_mask = (txt_input != self._txt_pad_token).unsqueeze(1) # B x 1 x S
        
        txt_input = batch.src_txt
        txt_length = batch.src_length
        txt_mask = batch.src_txt_mask

        gls_input = batch.gloss
        gls_length = batch.gloss_length

        # Target data related
        # trg_input = batch.target[0] # B x S x trg_dim
        # trg_length = batch.target[1]
        # trg_mask = (trg_input != self._trg_pad_token)[:, :, 0].unsqueeze(1) # B x 1 x S

        trg_input = batch.trg_seq
        trg_length = batch.trg_length
        trg_mask = batch.trg_seq_mask

        # dec_out: B x S x skel_dim
        # gls_prob: B x S x gls_vocab_dim
        dec_out, gls_prob = self.forward(
                txt_input=txt_input,
                txt_mask=txt_mask,
                txt_length=txt_length,
                trg_input=trg_input,
                trg_mask=trg_mask,
                gls_input=gls_input,
        )

        if self.do_translation:
            assert gls_prob is not None
            
            # Get translation loss
            translation_loss = translation_loss_function(
                pred=gls_prob,
                target=gls_input,
            ) * translation_loss_weight
        
        skeleton_output, _, _, _ = dec_out # B x S x dim

        # Caculate generation loss
        generation_loss = generation_loss_function(
            pred=skeleton_output,
            target=trg_input,
        ) * generation_loss_weight

        return translation_loss, generation_loss


def build_model(
    config: dict,
    txt_vocab,
    gls_vocab,
    do_translation=True,
    do_generation=True,
):
    
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    '''
    Build encoder part.
    '''
    # Word embeddings layer
    txt_emb = Embeddings(
        num_heads=config['encoder']['num_heads'],
        vocab_size=len(txt_vocab),
        padding_idx=txt_padding_idx,
        **config['encoder']['embeddings']
    )
    # Build encoder
    enc_dropout = config['encoder'].get('dropout', 0.0)
    enc_emb_dropout = config['encoder']['embeddings'].get('dropout', enc_dropout)
    if config['encoder']['type'] == 'transformer':
            assert (
                config['encoder']['embeddings']['embedding_dim']
                == config['encoder']['hidden_size']), \
                    'For transformer, emb_size must be same as hidden_size.'
            # Transformer encoder
            encoder = TransformerEncoder(
                emb_size=txt_emb.embedding_dim,
                emb_dropout=enc_emb_dropout,
                **config['encoder']
            )
    else:
        # Recurrent encoder
        raise NotImplementedError

    if do_translation:
        gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab))

    '''
    Build decoder part.
    '''
    # skeleton embedding layers (210 -> 512)
    target_size = config['decoder']['target_size']
    emb_dim = config['decoder']['embeddings']['embedding_dim']
    trg_emb = nn.Linear(target_size, emb_dim, bias=False)
    
    if do_generation:
        dec_emb_dropout = config['decoder']['dropout']
        if config['decoder']['type'] == 'transformer':
            decoder = TransformerDecoder(
                **config['decoder'],
                encoder=encoder,
                emb_dropout=dec_emb_dropout,
            )
    else:
        decoder = None

    # Build encoder and decoder model
    model = SignGenModel(
        encoder=encoder,
        decoder=decoder,
        txt_embed=txt_emb,
        txt_vocab=txt_vocab,
        trg_embed=trg_emb,
        gls_vocab=gls_vocab,
        gloss_output_layer=gloss_output_layer,
    )

    # custom initialization of model parameters
    initialize_model(model, config, txt_padding_idx)

    return model