import argparse
import os
import opts

from utils import *
from opts import *
from PIL import Image, ImageDraw, ImageFont


def render_head(Vp_x, Vp_y, draw, width=1):
    # draw_line(draw, Vp_x, Vp_y, NECK, width, fill='red')
    # draw_line(draw, Vp_x, Vp_y, HEAD, width, fill='red')
    draw_line(draw, Vp_x, Vp_y, LEFT_ARM, width, fill='red')
    draw_line(draw, Vp_x, Vp_y, RIGHT_ARM, width, fill='red')


def render_face(Vf_x, Vf_y, draw, width=1):
    # Draw faceline
    draw_line(draw, Vf_x, Vf_y, FACE_LINE, width, fill='white')
    # Draw left and right eyebrow
    draw_line(draw, Vf_x, Vf_y, LEFT_EYEBROW, width, fill='white')
    draw_line(draw, Vf_x, Vf_y, RIGHT_EYEBROW, width, fill='white')
    # Draw nose
    draw_line(draw, Vf_x, Vf_y, NOSE, width, fill='white')
    draw_line(draw, Vf_x, Vf_y, NOSE_BOTTOM, width, fill='white')
    # Draw left and right eye
    draw_line(draw, Vf_x, Vf_y, LEFT_EYE, width, is_polygon=True, fill='white')
    draw_line(draw, Vf_x, Vf_y, RIGHT_EYE, width, is_polygon=True, fill='white')
    # Draw mouth
    draw_line(draw, Vf_x, Vf_y, OUTER_MOUTH, width, is_polygon=True, fill='white')
    draw_line(draw, Vf_x, Vf_y, INNER_MOUTH, width, is_polygon=True, fill='white')
    # Draw left and right Pulpil
    # draw_circle(draw, Vf_x[PULPIL[0]], Vf_y[PULPIL[0]], fill='white', r=width)
    # draw_circle(draw, Vf_x[PULPIL[1]], Vf_y[PULPIL[1]], fill='white', r=width)


def get_blank_img(clr, width, height):
    blank_img = Image.new(clr, (width, height))
    draw = ImageDraw.Draw(blank_img)

    return blank_img, draw


def _get_parser():
    parser = argparse.ArgumentParser()
    opts.render_opts(parser)
    args = parser.parse_args()
    
    return args


# def main():
#     # Get arguments
#     args = _get_parser()

#     # Get required file list
#     paths = sorted(get_file_list(os.path.join(args.data, '*.json')))
#     renders = sorted(get_file_list(os.path.join(args.rendered, '*.png')))
#     assert len(paths) == len(renders), \
#         'Skeleton jsons and rendered images not same.'
    
#     for p, r in zip(paths, renders):
#         data_json = read_json(p)
#         person_info = data_json['people'][0] # Get the first person detected
#         Vf_x, Vf_y = get_Vx_Vy(person_info['face_keypoints_2d'])
#         Vp_x, Vp_y = (person_info['pose_keypoints_2d']) 
#         # Set a blank images with width and height value from the rendered image
#         rendered_img = Image.open(r)
#         w, h = rendered_img.size
#         # Create a blank image
#         image = Image.new('RGB', (w, h))
#         draw = ImageDraw.Draw(image)
        
#         if args.face:
#             # Draw face part
#             render_face(Vf_x, Vf_y, draw)
#         if args.head:
#             # Draw HEAD part
#             render_head(Vp_x, Vp_y, draw)
        
#         image.save('./test.png')

# if __name__ == '__main__':
#     main()