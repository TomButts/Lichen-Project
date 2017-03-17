from skimage.transform import rotate, rescale
import matplotlib.pyplot as plt
from random import uniform, randint, choice
from skimage.io import imread
import numpy as np

def transform(image_path):
    image = imread(image_path)

    # random flip
    if choice([True, False]):
        image = np.fliplr(image)

    # rotate
    degrees = uniform(0, 360)

    image = rotate(image, degrees)

    # scale
    scale = uniform(0.75, 1.25)

    image = rescale(image, scale)

    return image
