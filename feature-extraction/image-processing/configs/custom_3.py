options = dict(
    grey_required = True,
    glcm = dict(
        modes=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
    ),
    orb = dict(
        downscale=1.2,
        n_scales=8,
        n_keypoints=85,
        fast_n=9,
        fast_threshold=0.08,
        harris_k=0.04
    ),
    kmeans = dict(
        clusters = 2,
    ),
    glcm_func = dict(
        distances = [1],
        angles = [0],
        levels = 256,
        symmetric = True,
        normed = True
    )
)

# features.append(orb_extractor.keypoints.tolist())
