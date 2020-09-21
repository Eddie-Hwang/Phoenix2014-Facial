import opts
import pandas as pd
import os
import argparse
import glob

from tqdm import tqdm
from utils import *
from multiprocessing import Pool


def _get_parser():
    parser = argparse.ArgumentParser()
    opts.process_opts(parser)
   
    return parser.parse_args()

def make_dataset(corpus, d_type, keypts_path):
    row, _ = corpus.shape
    data = dict()
    for i in tqdm(range(row)):
        tr_name = corpus.loc[i, 'name']
        k_path = os.path.join(keypts_path, '{}/{}/*.json'.format(d_type, tr_name))
        tr_keypts = glob.glob(k_path)
        if len(tr_keypts):
            '''
            We make dataset format as below.
            TODO:
                The keypoints only contain seqeunce.
                There is no exact time stamp.
            '''
            data[tr_name] = {
                'text': corpus.loc[i, 'translation'],
                'gloss': corpus.loc[i, 'orth'],
                'face_keypoints_2d': [],
                'pose_keypoints_2d': [],
            }
            # Add face and pose keypoints
            for keypt in tr_keypts:
                try:
                    data_json = read_json(keypt)
                except:
                    pass
                keypts_info = data_json['people'][0]
                # check data quality
                condition_1 = [0]*len(keypts_info['face_keypoints_2d']) != keypts_info['face_keypoints_2d']
                condition_2 = [0]*len(keypts_info['pose_keypoints_2d']) != keypts_info['pose_keypoints_2d']
                if condition_1 and condition_2:
                    data[tr_name]['face_keypoints_2d'].append(keypts_info['face_keypoints_2d'])
                    data[tr_name]['pose_keypoints_2d'].append(keypts_info['pose_keypoints_2d'])
    
    return data
    
def main():
    args = _get_parser()
    
    # Read sign language corpus
    tr_corpus = pd.read_csv(os.path.join(args.annotation, 'PHOENIX-2014-T.train.corpus.csv'), sep='|')
    dev_corpus = pd.read_csv(os.path.join(args.annotation, 'PHOENIX-2014-T.dev.corpus.csv'), sep='|')
    test_corpus = pd.read_csv(os.path.join(args.annotation, 'PHOENIX-2014-T.test.corpus.csv'), sep='|')

    # Make train, dev and test dataset
    tr_data = make_dataset(tr_corpus, 'train', args.keypts)
    dev_data = make_dataset(dev_corpus, 'dev', args.keypts)
    test_data = make_dataset(test_corpus, 'test', args.keypts)

    # Save dataset
    if args.save_pickle:
        save_pickle(os.path.join(args.processed, 'train.pickle'), tr_data)
        save_pickle(os.path.join(args.processed, 'dev.pickle'), dev_data)
        save_pickle(os.path.join(args.processed, 'test.pickle'), test_data)
    if args.save_json:
        save_json(os.path.join(args.processed, 'train.json'), tr_data)
        save_json(os.path.join(args.processed, 'dev.json'), dev_data)
        save_json(os.path.join(args.processed, 'test.json'), test_data)

if __name__ == '__main__':
    main()
