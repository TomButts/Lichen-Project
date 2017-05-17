options = dict(
    grey_required = True,
    glcm = dict(
        modes=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
    ),
    kmeans = dict(
        clusters = 5,
    ),
    glcm_func = dict(
        distances = [1],
        angles = [0],
        levels = 256,
        symmetric = True,
        normed = True
    )
)
