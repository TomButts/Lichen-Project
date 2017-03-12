# Main script that will call feature extraction methods and collect output into
# a feature and target file
import glcm
from skimage import data

# use example of glcm features
# image = data.camera()
#
# modes = ['dissimilarity', 'correlation', 'homogeneity']
#
# # grass in this example
# patches = [ [(474, 291), (494, 311)],  [(440, 433), (460, 453)] ]
#
# grass_features = glcm.glcm_features(image, modes, patches)
