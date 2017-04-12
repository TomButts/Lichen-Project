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
        fast_threshold=0.08,
        harris_k=0.04
    ),
    kmeans = dict(
        clusters = 5,
    ),
    transform_factor=2
)
