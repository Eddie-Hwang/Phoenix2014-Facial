import opts
import pandas as pd
import os
import argparse
import glob
import numpy as np

from tqdm import tqdm
from utils import *
from pca import pca

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='./PHOENIX-2014-T_FACE')
    parser.add_argument('-n_components', type=int, default=10)
    
    return parser.parse_args()


def make_dataset(corpus_path, landmark_path, estimator):
    corpus = pd.read_csv(corpus_path, sep='|')
    row, _ = corpus.shape

    if estimator is not None:
        data = {'estimator': estimator}
    else:
        data = dict()
    
    for i in tqdm(range(row)):
        sl_name = corpus.loc[i, 'name']
        
        # Get landmark list 
        landmark_list = sorted(glob.glob(os.path.join(landmark_path, '{}/*_0.json'.format(sl_name))))
        filpped_landmark_list = sorted(glob.glob(os.path.join(landmark_path, '{}/*_1.json'.format(sl_name))))
        
        transformed_list = list()
        transformed_flipped_list = list()

        for l, fl in zip(landmark_list, filpped_landmark_list):
            _tmp_l = read_json(l)
            _tmp_fl = read_json(fl)

            if _tmp_l is not None:
                if estimator is not None:
                    _l = np.array(_tmp_l).reshape(1, -1)
                    transformed_l = list(estimator.transform(_l)[0])
                else:
                    transformed_l = _tmp_l
            
            if _tmp_fl is not None:
                if estimator is not None:
                    _fl = np.array(_tmp_fl).reshape(1, -1)
                    transformed_fl = list(estimator.transform(_fl)[0])
                else:
                    transformed_fl = _tmp_fl

            transformed_list.append(transformed_l)
            transformed_flipped_list.append(transformed_fl)
        
        if len(transformed_list) > 0 and len(transformed_flipped_list) > 0:
            # Create non-flipped data dict
            data["{}_0".format(sl_name)] = {
                'text': corpus.loc[i, 'translation'],
                'gloss': corpus.loc[i, 'orth'],
                'landmark': transformed_list,
            }
            # Create flipped data dict
            data["{}_1".format(sl_name)] = {
                'text': corpus.loc[i, 'translation'],
                'gloss': corpus.loc[i, 'orth'],
                'landmark': transformed_flipped_list,
            }
        
    print('{} has {} number of samples.'.format(corpus_path, len(data)-1))

    return data


def main():
    args = _get_parser()

    # Set annotation dataset path
    tr_corpus_path = os.path.join(args.data, 'annotation/PHOENIX-2014-T.train.corpus.csv')
    dev_corpus_path = os.path.join(args.data, 'annotation/PHOENIX-2014-T.dev.corpus.csv')
    tst_corpus_path = os.path.join(args.data, 'annotation/PHOENIX-2014-T.test.corpus.csv')
    _corpus = [tr_corpus_path, dev_corpus_path, tst_corpus_path]

    # Set facial landmark path
    tr_landmarks = os.path.join(args.data, 'train')
    dev_landmarks = os.path.join(args.data, 'dev')
    tst_landmarks = os.path.join(args.data, 'test')
    _landmark = [tr_landmarks, dev_landmarks, tst_landmarks]

    # Get PCA
    dir_list = glob.glob(os.path.join(tr_landmarks, '*'))
    all_landmarks = list()
    for _dir in dir_list:
        landmark_paths = sorted(glob.glob(os.path.join(_dir, '*.json')))
        all_landmarks.extend(landmark_paths)
    
    if args.n_components != 0:
        landmarks = np.array(
            [read_json(landmark) for landmark in tqdm(all_landmarks[:]) if read_json(landmark) is not None]
        )
        e = pca(
            X=landmarks,
            n_pc=args.n_components,
        )
    else:
        e = None

    for c, l in zip(_corpus, _landmark):
        _data = make_dataset(c, l, e)
        _type = os.path.split(l)[-1]
        
        save_pickle(
            path=os.path.join(args.data, '{}.pickle'.format(_type)),
            data=_data
        )


if __name__ == '__main__':
    main()
 