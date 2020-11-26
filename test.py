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
import numpy as np
import glob

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-config',
        default='./model_configs/default.yaml',
        type=str,
        help='Traning configuration file (yaml).'
    )
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='./sign_model/best.ckpt')
    parser.add_argument('--output', type=str, default='./sign_model/output')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--learned_pca', default='./pca_result/learned_pca.pickle')
    parser.add_argument('--test_path', default='./PHOENIX-2014-T_preprocessed/test')
    
    args = parser.parse_args()

    return args

def draw_landmark(args, x_cor, y_cor):
    image, draw = get_blank_img(
        clr='L', 
        width=args.img_size, 
        height=args.img_size
    )
    # Render facial expression
    render_face(x_cor, y_cor, draw, width=1)

    return image

def test(args):
    configs = load_config(args.config)
    ckpt_path = args.ckpt_path
    # Load data and vocabulary
    tr_data, dev_data, tst_data, txt_vocab, gls_vocab = load_data(configs['data'])
    
    if not(os.path.exists('./sign_model/test_result.pickle')):
        # Test
        result = test_on_data(
            config=configs,
            ckpt_path=ckpt_path,
            tst_data=tst_data,
            txt_vocab=txt_vocab,
            gls_vocab=gls_vocab,
        )
        # Temperory save result
        save_pickle('./sign_model/test_result.pickle', result)
        print('[INFO] Test_loss: {}'.format(result['valid_generation_loss']))
        print('[INFO] saved.')
    else:
        result = load_pickle('./sign_model/test_result.pickle')

    # Check output directory
    if not(os.path.exists(args.output)):
        os.mkdir(args.output)

    # Load learned PCA
    estimator = load_pickle(args.learned_pca)

    # Save rendered image
    for scene_num, (skel_hyp, skel_ref) in enumerate(tqdm(zip(result['skel_hyp'], result['skel_ref']))):
        # Get video information
        vid_name = result['vid_name'][scene_num]   
        txt = result['txt_ref'][scene_num]      
        
        # PCA inverse transform
        landmarks_hyp = estimator.inverse_transform(skel_hyp)
        landmarks_ref = estimator.inverse_transform(skel_ref)

        n_samples, dim = landmarks_ref.shape

        # Get x and y coordinates of predicted landmarks
        x_cors_hyp = np.array([landmarks_hyp[:, i] for i in range(0, dim, 2)]).T
        y_cors_hyp = np.array([landmarks_hyp[:, i+1] for i in range(0, dim, 2)]).T + 5

        x_cors_ref = np.array([landmarks_ref[:, i] for i in range(0, dim, 2)]).T
        y_cors_ref = np.array([landmarks_ref[:, i+1] for i in range(0, dim, 2)]).T + 5

        # Create base image
        base_width = args.img_size * n_samples
        base_height = args.img_size * 3 + 100
        base_img = Image.new(
            'RGB', 
            (base_width, base_height)
        )
        base_draw = ImageDraw.Draw(base_img)
        base_draw.text(
            xy=(base_width/2, base_height - 50), 
            text=txt, 
            fill='white', 
            font=ImageFont.truetype("DejaVuSans.ttf", 40)
        )

        # Get gt face image path
        gts = sorted(glob.glob(
            os.path.join(args.test_path, '{}/face_*.png'.format(vid_name))
        ))

        assert len(gts) == n_samples

        for i in range(n_samples):
            # Get predicted face
            image_hyp = draw_landmark(args, x_cors_hyp[i], y_cors_hyp[i])
            # Paste the rendered image on the base image
            base_img.paste(image_hyp, (i*args.img_size, 0))

            # Get ground truth landmarks
            image_ref = draw_landmark(args, x_cors_ref[i], y_cors_ref[i])
            # Paste the rendered image on the base image
            base_img.paste(image_ref, (i*args.img_size, args.img_size * 1))

            # Get ground truth face
            gt_face = Image.open(gts[i])
            base_img.paste(gt_face, (i*args.img_size, args.img_size * 2))

        # Save final output image
        base_img.save('{}/{}.png'.format(args.output, vid_name))
        

def main():
    args = _get_parser()
    
    # Allocating to selected gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    test(args=args)

if __name__ == '__main__':
    main()

