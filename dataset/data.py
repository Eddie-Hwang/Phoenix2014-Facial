from dataset.dataset import SignProductionDataset
from torchtext import data
from utils import *
from opts import *
from dataset.vocab import build_vocab

import os
import torch

def normalization(features):
    raise NotImplementedError



def load_data(args):

    vid_id_field = data.RawField()

    txt_field = data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=True,
        include_lengths=True,
    )

    gls_field = data.Field(
        pad_token=PAD_TOKEN,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    target_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        # preprocessing=normalization, #TODO
        batch_first=True,
        include_lengths=True,
        pad_token=torch.zeros((PAD_FEATURE_SIZE,)),
    )

    # Load processed meta data
    _tr_data = load_pickle(os.path.join(args.data, 'train.pickle'))
    _dev_data = load_pickle(os.path.join(args.data, 'dev.pickle'))
    _tst_data = load_pickle(os.path.join(args.data, 'test.pickle'))

    # Build train dataset
    tr_data = SignProductionDataset(
        _file=_tr_data,
        fields=(vid_id_field, txt_field, gls_field, target_field)
    )

    # Build vocab and update these to the fields
    txt_vocab, gls_vocab = build_vocab(_tr_data)
    txt_field.vocab = txt_vocab
    gls_field.vocab = gls_vocab

    # Build dev dataset
    dev_data = SignProductionDataset(
        _file=_tr_data,
        fields=[vid_id_field, txt_field, gls_field, target_field]
    )

    # Build test dataset
    tst_data = SignProductionDataset(
        _file=_tr_data,
        fields=[vid_id_field, txt_field, gls_field, target_field]
    )

    return tr_data, dev_data, tst_data, txt_vocab, gls_vocab


def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)


def make_data_iter(
    dataset,
    batch_size,
    batch_type,
    train=False,
    shuffle=False,
):
    """
    Returns a torchtext iterator for a torchtext dataset.
    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """
    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: data.interleave_keys(len(x.txt), len(x.target)),
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