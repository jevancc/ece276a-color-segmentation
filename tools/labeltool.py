import os
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
from roipoly import MultiRoi

parser = argparse.ArgumentParser(description='Label stop sign image')
parser.add_argument('-i',
                    nargs=1,
                    help='input image path',
                    dest='input',
                    required=True)
args = parser.parse_args()

IMG_FILE = args.input[0]

img = cv2.imread(IMG_FILE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

COLORS = [('COLOR_STOP_SIGN_RED', 'red'), ('COLOR_OTHER_RED', 'orangered'),
          ('COLOR_BROWN', 'brown'), ('COLOR_ORANGE', 'orange'),
          ('COLOR_BLUE', 'cyan'), ('COLOR_OTHER', 'black')]
rois = []
color_pixels = {}
all_color_mask = np.zeros(img.shape[:-1])


def prompt_is_ok(msg):
    print(msg, end='')
    answer = input()
    return answer == 'y'


for color, roi_color in COLORS:
    is_ok = False
    if not prompt_is_ok(f'Do you want to label color {color}? [y/n]: '):
        color_pixels[color] = np.array([])
        continue

    while not is_ok:
        print(f'Labeling color {color} ...')
        fig = plt.figure()
        plt.imshow(img, interpolation='nearest', cmap='Greys')
        plt.title(f'Add ROI for Color:{color}')

        multiroi = MultiRoi(color_cycle=(roi_color,))

        tmask = np.zeros(img.shape[:-1])
        for name, roi in multiroi.rois.items():
            mask = roi.get_mask(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            tmask += mask

        masked_img = img.copy()
        masked_img[tmask == 0, :] = 0
        plt.imshow(masked_img)
        plt.show()
        is_ok = prompt_is_ok(f'Is color {color} labeled correctly? [y/n]: ')
        if not is_ok:
            print(f'Please label color {color} again.')
        else:
            rois.extend(multiroi.rois.values())
            pixels = img[tmask != 0, :].reshape(-1, 3)
            color_pixels[color] = pixels

data = {**color_pixels}

if prompt_is_ok(f'Do you want to label stop signs region? [y/n]: '):
    is_ok = False
    while not is_ok:
        print(f'Labeling stop signs ...')
        fig = plt.figure()
        plt.imshow(img, interpolation='nearest', cmap='Greys')
        plt.title(f'Add ROI for stop signs')

        multiroi = MultiRoi(color_cycle=('g',))

        tmask = np.zeros(img.shape[:-1])
        for name, roi in multiroi.rois.items():
            mask = roi.get_mask(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            tmask += mask

        masked_img = img.copy()
        masked_img[tmask == 0, :] = 0
        plt.imshow(masked_img)
        plt.show()
        is_ok = prompt_is_ok(f'Are stop signs labeled correctly? [y/n]: ')
        if not is_ok:
            print(f'Please label stop signs again.')
        else:
            rois.extend(multiroi.rois.values())
            tmask[tmask != 0] = 1
            stop_sign_mask = tmask
            stop_sign_roi = multiroi

    data['MASK_STOP_SIGN'] = stop_sign_mask
else:
    data['MASK_STOP_SIGN'] = np.zeros(img.shape[:-1])

img_name = os.path.splitext(IMG_FILE)[0]
plt.figure()
plt.imshow(img)
for roi in rois:
    roi.display_roi()
plt.axis('off')
plt.savefig(f'{img_name}-roi.png', bbox_inches='tight')
plt.title('Labeling Result')
plt.show()

np.savez(f'{img_name}.npz', **data)
