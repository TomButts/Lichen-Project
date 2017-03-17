import configs.example1 as config
import glcm
from skimage.feature import ORB
from skimage.color import rgb2grey
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import flatten

def features(image_path):
    """Extracts a set of features (described in config file) from an image.
    Args:
        image_path: the path to the image

    Return:
        an array of features which depending on the config options
    """

    features = []

    image = imread(image_path)

    if config.grey_required:
        grey_image = rgb2grey(image)

    # GLCM features
    if config.glcm:
        glcm_config = config.glcm

        glcm_features = glcm.glcm_features(grey_image, glcm_config['modes'])

        features.append(glcm_features)

    # ORB features
    if config.orb:
        orb_config = config.orb

        orb_extractor = ORB(
            downscale=orb_config['downscale'],
            n_scales=orb_config['n_scales'],
            n_keypoints=orb_config['n_keypoints'],
            fast_n=orb_config['fast_n'],
            fast_threshold=orb_config['fast_threshold'],
            harris_k=orb_config['harris_k'])

        orb_extractor.detect_and_extract(grey_image)

        features.append(orb_extractor.keypoints.tolist())

        # features.append(orb_extractor.descriptors.tolist())

    return list(flatten.flatten(features))