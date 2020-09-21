from dataset.data import load_data
from model.model import build_model
from train_manager import TrainManager
from utils import *
from opts import *
from model.prediction import test_on_data
from render import render_face, get_blank_img
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import argparse
import os, sys


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-config',
        default='./model_configs/default.yaml',
        type=str,
        help='Traning configuration file (yaml).'
    )    
    parser.add_argument('--gpu_id', type=str, default='0')

    return parser.parse_args()


def train(args):
    configs = load_config(args.config)
    # Load data and vocabulary
    tr_data, dev_data, tst_data, txt_vocab, gls_vocab = load_data(configs['data'])
    
    do_translation = configs['training'].get('translation_loss_weight', 0.0) > 0.0
    do_generation = configs['training'].get('generation_loss_weight', 1.0) > 0.0
    
    # Build model
    model = build_model(
        config=configs['model'],
        txt_vocab=txt_vocab,
        gls_vocab=gls_vocab,
        do_translation=do_translation,
        do_generation=do_generation,
    )

    # Train manager
    trainer = TrainManager(
        model=model,
        config=configs,
    )

    # Start training and validation
    trainer.train_and_validation(
        train_data=tr_data,
        valid_data=dev_data,
    )
    
    del tr_data, dev_data


def main():
    args = _get_parser() 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    train(args)


if __name__ == '__main__':
    main()