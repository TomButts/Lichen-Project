"""Example configuration file

For excluding a feature make the feature equal to None.
If the feature is used make sure all the required parameters are available.
Optional parameters can be handled as None.
"""

options = dict(
    grey_required = True,
    glcm = dict(
        modes=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
    ),
    orb = dict(
        downscale=1.2,
        n_scales=8,
        n_keypoints=50,
        fast_n=9,
        fast_threshold=0.12,
        harris_k=0.14
    ),
    kmeans = dict(
        clusters = 5,
    )
)
