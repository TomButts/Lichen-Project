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
    ),
    glcm_func = dict(
        distances = [7],
        angles = [0],
        levels = 256,
        symmetric = True,
        normed = True
    )
)

# features.append(orb_extractor.keypoints.tolist())
# features.append(orb_extractor.scales.tolist())
# features.append(orb_extractor.orientations.tolist())
# features.append(orb_extractor.responses.tolist())
