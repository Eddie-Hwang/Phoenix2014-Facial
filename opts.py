import argparse

def render_opts(parser):
    ''' set arguments of render.py '''
    parser.add_argument('-data', default='./test/keypts')
    parser.add_argument('-rendered', default='./test/render')
    parser.add_argument('-face', type=bool, default=True)
    parser.add_argument('-head', type=bool, default=False)

def process_opts(parser):
    ''' Set arguments of process.py '''
    parser.add_argument('-keypts', default='./data/keypts')
    parser.add_argument('-annotation', default='./data/manual')
    parser.add_argument('-processed', default='./data')
    parser.add_argument('-save_pickle', type=bool, default=True)
    parser.add_argument('-save_json', type=bool, default=False)

def dataset_opts(parser):
    ''' Set vocab arguments '''
    parser.add_argument('-data', default='./data')

'''
Token to be used
'''
SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

PAD_FEATURE_SIZE = 210


''' 
Skeleton part description 
'''
FACE_LINE = [i for i in range(17)]
LEFT_EYEBROW = [i for i in range(17, 22)]
RIGHT_EYEBROW = [i for i in range(22, 27)]

NOSE = [i for i in range(27, 31)]
NOSE_BOTTOM = [i for i in range(31, 36)]

LEFT_EYE = [i for i in range(36, 42)]
RIGHT_EYE = [i for i in range(42, 48)]

OUTER_MOUTH = [i for i in range(48, 60)]
INNER_MOUTH = [i for i in range(60, 68)]

PULPIL = [68, 69]

NECK = [0, 1]
HEAD = [17, 15, 0, 16, 18]