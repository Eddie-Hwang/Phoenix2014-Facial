import json
import glob
import pickle
import yaml
import torch.nn as nn


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training
    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def load_config(path):
    '''
    Load yaml configuration file.
    '''
    with open(path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs

def load_pickle(path):
    ''' Load pickle file '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(path, data):
    ''' Save pickle file '''
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def save_json(path, data):
    ''' Save json file '''
    with open(path, 'w') as f:
        json.dump(data, f)

def read_json(path):
    ''' Load json file '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_file_list(path):
    ''' Get file list '''
    return glob.glob(path)

def get_Vx_Vy(val):
    ''' 
    Get X and Y points. 
    Openpose gives list [x1,y1,c1,x2,y2,c2,...,xn,yn,cn]
    Here, we only need x and y value.

    Args:
        val: extracted keypoints list from Openpose
    '''
    Vx = list()
    Vy = list()

    n = len(val)
    for i in range(0, n, 3):
        Vx.append(val[i])
        Vy.append(val[i+1])
    
    return Vx, Vy

def draw_line(draw, Vx, Vy, idxs, width, fill="red", is_polygon=False):
    '''
    Draw line on the given image
    Args:
        draw: draw object image
        Vx: list of x point
        Vy: list of y point
        idxs: the given indexes of point
        fill: line color
        width: line width
        draw_polygon: 
    '''
    for i in range(len(idxs)-1):
        start = (Vx[idxs[i]], Vy[idxs[i]])
        end = (Vx[idxs[i+1]], Vy[idxs[i+1]])
        draw.line([start, end], fill=fill, width=width)
    if is_polygon:
        # Draw last line
        start = (Vx[idxs[0]], Vy[idxs[0]]) 
        end = (Vx[idxs[-1]], Vy[idxs[-1]]) 
        draw.line([start, end], fill=fill, width=width)
    


    # if not(is_polygon):
    #     for idx in idxs:
    #         start = (Vx[idx], Vy[idx])
    #         end = (Vx[idx+1], Vy[idx+1])
    #         draw.line([start, end], fill=fill, width=width)
    # else:
    #     for idx in idxs:
    #         start = (Vx[idx], Vy[idx])
    #         end = (Vx[idx+1], Vy[idx+1])
    #         draw.line([start, end], fill=fill, width=width)
    #     # Draw last line
    #     start = (Vx[idxs[0]], Vy[idxs[0]]) 
    #     end = (Vx[idxs[-1]], Vy[idxs[-1]]) 
    #     draw.line([start, end], fill=fill, width=width)

def draw_circle(draw, x, y, fill, r):
    leftup = (x-r, y-r)
    rightup = (x+r, y+r)
    draw.ellipse([leftup, rightup], fill)
    




