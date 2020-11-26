from dataset.dataset import SignProductionDataset
from torchtext import data
from utils import *
from opts import *
from model.vocab import *

import os
import torch
import sys
import numpy as np


# def normalization(features):
#     processed = list()
#     for i in range(0, len(features), SAMPLING_RATE):
#         feature = features[i]
#         x_cor, y_cor = get_Vx_Vy(feature)
#         new_x_cor, new_y_cor = relocate_and_scaling(
#             x_cor=np.array(x_cor), 
#             y_cor=np.array(y_cor),
#             center_idx=71, # Neck index
#             size=PREPROCESSING_SIZE, 
#             scale=PREPROCESSING_SCALE,
#         )
#         # We need face skeleton points only
#         new_feature = new_x_cor.tolist()[:70] + new_y_cor.tolist()[:70]
#         processed.append(new_feature)
#         # new_feature = new_x_cor.tolist() + new_y_cor.tolist()
#         # processed.append(new_feature)
    
#     return processed

def landmark_preprocessing(landmarks):
    '''
    Clipping the first and second PC by 0.5
    '''
    landmarks[:, 0] = 0.5
    landmarks[:, 1] = -0.5
    
    return landmarks


def load_data(data_configs):
    # Dataset path related
    data_path = data_configs['data_path']
    train_path = data_configs['train']
    dev_path = data_configs['dev']
    test_path = data_configs['test']
    
    # Text data related
    level = data_configs['level']
    txt_lowercase = data_configs['txt_lowercase']
    max_sent_length = data_configs.get('max_sent_length', -1)
    
    def tokenize_text(text):
        if level == "char":
            return list(text)
        else:
            return text.split()
    
    # Data fields
    vid_id_field = data.RawField()
    txt_field = data.Field(
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        tokenize=tokenize_text,
        fix_length=max_sent_length
        if max_sent_length != -1 else None,
        batch_first=True,
        lower=True,
        include_lengths=True,
    )
    gls_field = data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        batch_first=True,
        lower=False,
        tokenize=tokenize_text,
        include_lengths=True,
    )
    landmark_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        # preprocessing=landmark_preprocessing,
        preprocessing=None,
        batch_first=True,
        include_lengths=True,
        pad_token=torch.zeros((LANDMARK_SIZE,)),
    )

    # Load processed meta data
    train_data = load_pickle(train_path)
    dev_data = load_pickle(dev_path)
    tst_data = load_pickle(test_path)

    # vocab build related 
    gls_max_size = sys.maxsize
    gls_min_freq = 1
    txt_max_size = sys.maxsize
    txt_min_freq = 1
    
    # Build vocab dict
    gls_vocab = build_vocab(
        field='gls',
        min_freq=gls_min_freq,
        max_size=gls_max_size,
        dataset=train_data,
    )
    txt_vocab = build_vocab(
        field='txt',
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        dataset=train_data,
    )
    
    # Update vocab to the fields
    txt_field.vocab = txt_vocab
    gls_field.vocab = gls_vocab

    # Build train dataset
    tr_data = SignProductionDataset(
        _file=train_data,
        fields=[vid_id_field, txt_field, gls_field, landmark_field]
    )
    # Build dev dataset
    dev_data = SignProductionDataset(
        _file=dev_data,
        fields=[vid_id_field, txt_field, gls_field, landmark_field]
    )
    # Build test dataset
    tst_data = SignProductionDataset(
        _file=tst_data,
        fields=[vid_id_field, txt_field, gls_field, landmark_field]
    )

    return tr_data, dev_data, tst_data, txt_vocab, gls_vocab


def make_data_iter(
    dataset,
    batch_size,
    batch_type,
    train=False,
    shuffle=False,
):
    # batch_size_fn = token_batch_size_fn if batch_type == "token" else None
    batch_size_fn = None

    if train:
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: data.interleave_keys(len(x.text), len(x.landmark)),
            shuffle=shuffle,
        )
    else:
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter