from torchtext import data
from utils import *

class SignProductionDataset(data.Dataset):

    def __init__(self, _file, fields, **kwargs):

        ''' 
        We load the processed data

        Args:
            path: data path must be string,
            fields: list of field
                [
                    vid_id_field,
                    txt_field,
                    gls_field,
                    target_field
                ]
        '''
        samples = _file

        fields = [
            ('vid_name', fields[0]),
            ('txt', fields[1]),
            ('gls', fields[2]),
            ('target', fields[3])
        ]
        
        examples = list()
        for s in samples.keys():
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    data=[
                        s,
                        sample['text'],
                        sample['gloss'],
                        sample['face_keypoints_2d'],
                        # sample['face_keypoints_2d'] + sample['pose_keypoints_2d'],
                        ],
                    fields=fields
                )
            )

        super().__init__(examples, fields, **kwargs)
            