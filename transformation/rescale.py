from __future__ import division

from skimage.transform import rescale
from skimage.io import imread

def adjust_size(image_path, max_height=800, max_width=800):
    image = imread(image_path)

    height = image.shape[1]
    width = image.shape[0]

    if height > max_height or width > max_width:
        scale = min((max_width/width), (max_height/height))

        image = rescale(image, scale)

    return image
