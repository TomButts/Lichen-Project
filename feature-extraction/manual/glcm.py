from skimage.feature import greycomatrix, greycoprops

# regions is array
def glcm_features(image, modes = None, regions = None):
    patches = [image]

    if regions != None:
        patches = []

        for roi in regions:
            patches.append(image[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]])

    # got to have some modes
    if modes == None:
        modes = ['dissimilarity', 'correlation']

    index = 0

    features = []

    for patch in patches:
        patch_features = []

        # get the grey level co occurence matrix
        glcm = greycomatrix(patch, [5], [0], 256, symmetric = True, normed = True)

        for mode in modes:
            # analyse the matrix for feature data
            patch_features.append(greycoprops(glcm, mode)[0, 0])

        features.append(patch_features)

        index += 1

    print(features)

    return features
