from torchtext import data
from utils import *

import numpy as np


class SignProductionDataset(data.Dataset):

    def __init__(self, _file, fields, **kwargs):

        samples = _file

        fields = [
            ('id', fields[0]),
            ('text', fields[1]),
            ('gloss', fields[2]),
            ('landmark', fields[3]),
        ]
        
        examples = list()
        for s in samples.keys():
            if s != 'estimator':
                sample = samples[s]
                # S x dim
                landmarks = np.array(sample['landmark'])
                # body_skel_pts = np.array(sample['pose_keypoints_2d'])
                # new_skel_pts = np.concatenate((face_skel_pts, body_skel_pts), axis=-1)
                examples.append(
                    data.Example.fromlist(
                        data=[
                            s,
                            sample['text'],
                            sample['gloss'],
                            landmarks,
                        ],
                        fields=fields
                    )
                )

        super().__init__(examples, fields, **kwargs)
            