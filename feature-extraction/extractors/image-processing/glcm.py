from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2grey

def glcm_features(image, modes=None):
    """Handles the creation of the GLCM and the extraction of features
       from said matrix.

    Args:
        image: the array of pixel values representing the image
    Return:
        an array of GLCM features
    """

    # modes of analysis to apply to glcm matrix
    if modes is None:
        modes = ['dissimilarity', 'correlation']

    features = []

    # get the grey level co occurence matrix
    glcm = greycomatrix(image, [1], [0], 256, symmetric=True, normed=True)

    for mode in modes:
        # analyse the matrix for feature data
        features.append(greycoprops(glcm, mode)[0, 0])

    # print(features)

    return features
