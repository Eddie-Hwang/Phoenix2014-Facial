import argparse
import os
import opts

from utils import *
from opts import *
from PIL import Image, ImageDraw, ImageFont

def render_head(Vp_x, Vp_y, draw):
    draw_line(draw, Vp_x, Vp_y, NECK, 3, fill='red')
    draw_line(draw, Vp_x, Vp_y, HEAD, 3, fill='red')

def render_face(Vf_x, Vf_y, draw):
    # Draw faceline
    draw_line(draw, Vf_x, Vf_y, FACE_LINE, 3, fill='white')
    # Draw left and right eyebrow
    draw_line(draw, Vf_x, Vf_y, LEFT_EYEBROW, 3, fill='white')
    draw_line(draw, Vf_x, Vf_y, RIGHT_EYEBROW, 3, fill='white')
    # Draw nose
    draw_line(draw, Vf_x, Vf_y, NOSE, 3, fill='white')
    draw_line(draw, Vf_x, Vf_y, NOSE_BOTTOM, 3, fill='white')
    # Draw left and right eye
    draw_line(draw, Vf_x, Vf_y, LEFT_EYE, 3, is_polygon=True, fill='white')
    draw_line(draw, Vf_x, Vf_y, RIGHT_EYE, 3, is_polygon=True, fill='white')
    # Draw mouth
    draw_line(draw, Vf_x, Vf_y, OUTER_MOUTH, 3, is_polygon=True, fill='white')
    draw_line(draw, Vf_x, Vf_y, INNER_MOUTH, 3, is_polygon=True, fill='white')
    # Draw left and right Pulpil
    draw_circle(draw, Vf_x[PULPIL[0]], Vf_y[PULPIL[0]], fill='white', r=3)
    draw_circle(draw, Vf_x[PULPIL[1]], Vf_y[PULPIL[1]], fill='white', r=3)

def _get_parser():
    parser = argparse.ArgumentParser()
    opts.render_opts(parser)
    args = parser.parse_args()
    
    return args

def main():
    # Get arguments
    args = _get_parser()

    # Get required file list
    paths = sorted(get_file_list(os.path.join(args.data, '*.json')))
    renders = sorted(get_file_list(os.path.join(args.rendered, '*.png')))
    assert len(paths) == len(renders), \
        'Skeleton jsons and rendered images not same.'
    
    for p, r in zip(paths, renders):
        data_json = read_json(p)
        person_info = data_json['people'][0] # Get the first person detected
        Vf_x, Vf_y = get_Vx_Vy(person_info['face_keypoints_2d'])
        Vp_x, Vp_y = get_Vx_Vy(person_info['pose_keypoints_2d']) 
        # Set a blank images with width and height value from the rendered image
        rendered_img = Image.open(r)
        w, h = rendered_img.size
        # Create a blank image
        image = Image.new('RGB', (w, h))
        draw = ImageDraw.Draw(image)
        
        if args.face:
            # Draw face part
            render_face(Vf_x, Vf_y, draw)
        if args.head:
            # Draw HEAD part
            render_head(Vp_x, Vp_y, draw)
        
        image.save('./test.png')

if __name__ == '__main__':
    main()