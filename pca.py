import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from utils import *
from render import *
from sklearn.decomposition import PCA


def plot_portraits(images, titles, h, w, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        # plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())


def pca(X, n_pc):
    estimator = PCA(
        n_components=n_pc,
        svd_solver='randomized',
        whiten=True,
    )
    estimator.fit(X)
    var_ratio = estimator.explained_variance_ratio_
    print('[INFO] {} number of components explain {:0.2f} of original dataset.'.format(
        n_pc, 
        np.sum(var_ratio)
        )
    )
    
    return estimator


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_path', default='./PHOENIX-2014-T_FACE/train')
    parser.add_argument('-n_components', type=int, default=10)
    parser.add_argument('-pca_results', default='./pca_result')

    return parser.parse_args()


def main():
    args = get_parser()
    # tr_path = os.path.join(args.save_path, 'dev')
    dir_list = glob.glob(os.path.join(args.save_path, '*'))
    
    all_landmarks = list()
    for _dir in dir_list:
        landmark_paths = sorted(glob.glob(os.path.join(_dir, '*.json')))
        all_landmarks.extend(landmark_paths)

    landmarks = np.array(
        [read_json(landmark) for landmark in tqdm(all_landmarks[:]) if read_json(landmark) is not None]
    )
    n_samples, dim = landmarks.shape
    print('number of samples: {}'.format(n_samples))
    
    # Get estimator
    estimator = pca(
            X=landmarks,
            n_pc=args.n_components,
    )
    # Save learned pickle
    save_pickle(
        path=os.path.join(args.pca_results, 'learned_pca.pickle'), 
        data=estimator,
    )
    print('Saved learned PCA at {}'.format(os.path.join(args.pca_results, 'learned_pca.pickle')))

    # Explore PCA subspace
    flatten = np.array([1 for _ in range(args.n_components)])
    pca_subspace_array = np.stack(
        arrays=[np.diagflat(flatten*i) for i in range(-4,5,2) if i != 0],
        axis=0,
    ).reshape(-1, 10)

    mean_face = estimator.inverse_transform(np.array([0 for _ in range(args.n_components)]))
    eigen_faces = estimator.inverse_transform(pca_subspace_array)

    if not(os.path.exists(args.pca_results)):
        os.mkdir(args.pca_results)

    # Draw and save pca mean face
    _m_dim = mean_face.shape[0]
    m_x_cor = np.array([mean_face[i] for i in range(0, _m_dim, 2)])
    m_y_cor = np.array([mean_face[i+1] for i in range(0, _m_dim, 2)])

    image = Image.new('L', (64, 64))
    draw = ImageDraw.Draw(image)
    render_face(m_x_cor, m_y_cor, draw)
    image.save(os.path.join(args.pca_results, 'mean_face.png'))

    # Draw and save pca subspace egien faces
    _, _dim = eigen_faces.shape
    x_cors = np.array([eigen_faces[:, i] for i in range(0, _dim, 2)]).T
    y_cors = np.array([eigen_faces[:, i+1] for i in range(0, _dim, 2)]).T
    
    all_images = list()
    for i, (x_cor, y_cor) in enumerate(zip(x_cors, y_cors)):
        image = Image.new('L', (64, 64))
        draw = ImageDraw.Draw(image)
        render_face(x_cor, y_cor, draw)
        image_path = os.path.join(args.pca_results, 'face_{}.png'.format(i))
        image.save(image_path)
        all_images.append(image_path)

    eigenfaces = [plt.imread(image) for image in all_images]
    plot_portraits(eigenfaces, None, 64, 64, 4, 10)
    plt.savefig(os.path.join(args.pca_results, 'subspace.png'))

    # Delete uneccessary files
    for image in all_images:
        os.remove(image)


if __name__=='__main__':
    # get_sample()
    main()
