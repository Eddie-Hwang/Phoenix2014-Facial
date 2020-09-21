import sys
import dlib
import cv2
import openface
import argparse
import glob
import os

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from utils import *

def alignment(img):
    # Read image
    image = cv2.imread(img)
    # Dectect face
    detected_faces = face_detector(image, 1)
    # print('Found {} faces in the image file {}'.format(len(detected_faces), img))

    # Check if there is no face detected
    if len(detected_faces) < 1:
        return
    
    # We only need the first face detected
    face_rect = detected_faces[0]

    # Detect facial landmarks

    landmarks = face_pose_predictor(image, face_rect)
    # assert landmarks.num_parts == 68
    if landmarks.num_parts != 68:
        return

    # Aligned face
    aligned_face = face_aligner.align(
        imgDim=image_size, 
        rgbImg=image,
        bb=face_rect,
        landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP,
        skipMulti=True,
    )
    # Here we also need flipped aligned face for data augmentation
    flipped_aligned_face = cv2.flip(aligned_face, 1)

    # Get landmark of aligned face
    landmark_list = get_landmarks(aligned_face)
    flipped_landmark_list = get_landmarks(flipped_aligned_face)

    # Identify data information
    data, _img = os.path.split(img)
    file_path, _name = os.path.split(data)
    _, _type= os.path.split(file_path)
    
    # Check directory
    folder_name = os.path.join(
        save_path, 
        _type, 
        _name
    )
    
    _img_name = _img.split('.')[0]
    # if landmark_list is not None:
    
    # save aligned face
    cv2.imwrite(
        os.path.join(folder_name, 'face_{}_0.png'.format(_img_name)), 
        aligned_face
    )
    save_json(
        os.path.join(folder_name, 'face_{}_0.json'.format(_img_name)),
        landmark_list
    )
    # data augmentation by flipping
    cv2.imwrite(
        os.path.join(folder_name, 'face_{}_1.png'.format(_img_name)),
        flipped_aligned_face
    )
    # save landmark json

    save_json(
        os.path.join(folder_name, 'face_{}_1.json'.format(_img_name)),
        flipped_landmark_list
    )



def get_landmarks(img_array):
    detected_faces = face_detector(img_array, 1)
    if len(detected_faces) < 1:
        return None
    face_rect = detected_faces[0]
    landmarks = face_pose_predictor(img_array, face_rect)
    landmark_list = list()
    for i in range(landmarks.num_parts):
        landmark_list.append(landmarks.part(i).x)
        landmark_list.append(landmarks.part(i).y)
    
    return landmark_list
    

def preprocessing_image(
    args,
    file_path,
    face_detector=None,
    face_pose_predictor=None,
    face_aligner=None,
):
    # multi-process
    data_list = glob.glob(os.path.join(file_path, '*'))
    
    for data in tqdm(data_list):
        # Get directory path
        _path, _dir_name = os.path.split(data)
        _, _type= os.path.split(_path)
        
        _full_path = os.path.join(
            args.save_path,
            _type,
            _dir_name,
        )
        
        # Create directory
        if not(os.path.exists(_full_path)):
            os.mkdir(_full_path)
        
        # Get image paths
        images = glob.glob(os.path.join(data, '*.*'))

        with Pool(processes=args.num_proc) as p:
            for i, img in enumerate(p.imap_unordered(alignment, images)):
                pass


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-predictor_model', default='/home/ejhwang/Phoenix2014-Facial/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('-dataset', default='./PHOENIX-2014-T/features/fullFrame-227x227px')
    parser.add_argument('-save_path', default='./PHOENIX-2014-T_FACE')
    parser.add_argument('-image_size', type=int, default=64)
    parser.add_argument('-num_proc', type=int, default=30)
    parser.add_argument('-landmarks', type=bool, default=True)

    return parser.parse_args()


def main():
    args = get_parser()

    if not(os.path.exists(args.save_path)):
        os.mkdir(args.save_path)
        os.mkdir(os.path.join(args.save_path, 'train'))
        os.mkdir(os.path.join(args.save_path, 'dev'))
        os.mkdir(os.path.join(args.save_path, 'test'))

    # Make global
    global face_detector
    global face_pose_predictor
    global face_aligner
    global image_size
    global save_path

    # Detect face
    face_detector = dlib.get_frontal_face_detector()
    # Detect facial landmarks
    face_pose_predictor = dlib.shape_predictor(args.predictor_model)
    # Face aligner using openface
    face_aligner = openface.AlignDlib(args.predictor_model)

    # Set image size
    image_size = args.image_size

    # Set save_path
    save_path = args.save_path

    tr_path = os.path.join(args.dataset, 'train')
    dev_path = os.path.join(args.dataset, 'dev')
    test_path = os.path.join(args.dataset, 'test')

    _paths = [tr_path, dev_path, test_path]
    for p in _paths:
        print(p)
        preprocessing_image(
            args=args,
            file_path=p,
            face_detector=face_detector,
            face_pose_predictor=face_pose_predictor,
            face_aligner=face_aligner,
        )


if __name__=='__main__':
    main()




    

