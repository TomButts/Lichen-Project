"""Example configuration file

For excluding a feature make the feature equal to None.
If the feature is used make sure all the required parameters are available.
Optional parameters can be handled as None.
"""

glcm = dict(
    modes = ['dissimilarity', 'correlation', 'homogeneity'],
)

orb = dict(
    downscale = 1.2,
    n_scales = 8,
    n_keypoints = 50,
    fast_n = 9,
    fast_threshold = 0.08,
    harris_k=0.04
)
