from skimage.feature import ORB
from skimage.color import rgb2grey
from skimage.io import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

import sys
import os

# Add any extra modules to the python path
import_path = os.path.abspath('./extractors/image-processing')
sys.path.append(import_path)

import glcm
import flatten

def feature_extractor(image_path, options=None):
    """Extracts a set of features (described in config file) from an image.
    Args:
        image_path: the path to the image
        options: the configuration file settings. In this case the settings should contain
                 relevant image processing feature options.
    Return:
        an array of features which depending on the config options
    """

    features = []

    image = imread(image_path)

    if 'grey_required' in options:
        grey_image = rgb2grey(image)

    # GLCM features
    if 'glcm' in options:
        glcm_config = options['glcm']

        glcm_features = glcm.glcm_features(grey_image, glcm_config['modes'])

        features.append(glcm_features)

    # ORB features
    if 'orb' in options:
        orb_config = options['orb']

        orb_extractor = ORB(
            downscale=orb_config['downscale'],
            n_scales=orb_config['n_scales'],
            n_keypoints=orb_config['n_keypoints'],
            fast_n=orb_config['fast_n'],
            fast_threshold=orb_config['fast_threshold'],
            harris_k=orb_config['harris_k'])

        orb_extractor.detect_and_extract(grey_image)

        # TODO add these to config system
        features.append(orb_extractor.keypoints.tolist())
        # features.append(orb_extractor.scales.tolist())
        # features.append(orb_extractor.orientations.tolist())
        # features.append(orb_extractor.responses.tolist())

        # features.append(orb_extractor.descriptors.tolist())

    if 'kmeans' in options:
        k_image = np.array(image, dtype=np.float64) / 255

        w, h, d = original_shape = tuple(k_image.shape)
        assert d == 3
        image_array = np.reshape(k_image, (w * h, d))

        kmeans = KMeans(n_clusters=options['kmeans']['clusters']).fit(image_array)

        features.append(kmeans.cluster_centers_.tolist())

    return list(flatten.flatten(features))
