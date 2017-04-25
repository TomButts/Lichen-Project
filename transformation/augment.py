from skimage.transform import rotate, rescale
import matplotlib.pyplot as plt
from random import uniform, randint, choice
from skimage.io import imread
import numpy as np

def augment(image_path):
    image = imread(image_path)

    # random flip
    if choice([True, False]):
        image = np.fliplr(image)

    # rotate
    degrees = choice([90, 180, 270])

    image = rotate(image, degrees, resize=True)

    # scale
    scale = uniform(0.6, 1.25)

    image = rescale(image, scale)

    return image
