from dataset.data import load_data
from model.model import build_model
from train_manager import TrainManager
from utils import *
from opts import *

import argparse
import os

def _get_parser():
    parser = argparse.ArgumentParser()
    dataset_opts(parser)
    parser.add_argument(
        '-config',
        default='./model_configs/default.yaml',
        type=str,
        help='Traning configuration file (yaml).'
    )
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()

    return args

def train(args):
    configs = load_config(args.config)

    # Load data and vocabulary
    tr_data, dev_data, tst_data, _txt_vocab, _gls_vocab = load_data(args)
    
    # Build model
    do_translation = configs['training'].get('translation_loss_weight', 1.0) > 0.0
    do_generation = configs['training'].get('generation_loss_weight', 1.0) > 0.0
    model = build_model(
        config=configs['model'],
        txt_vocab=_txt_vocab,
        gls_vocab=_gls_vocab,
        do_translation=do_translation,
        do_generation=do_generation,
    )

    trainer = TrainManager(
        model=model,
        config=configs,
    )

    trainer.train_and_validation(
        train_data=tr_data,
        valid_data=dev_data,
    )
    del tr_data, dev_data


def main():
    args = _get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    train(args=args)

if __name__ == '__main__':
    main()

