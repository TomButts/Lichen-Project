"""
Example configuration file

An example configuration file that will run the image processing feature
extractor on an ordered images directory. The labels file in this case
is a csv type file. The extra settings which the image processing extractor
requires should also be defined in this configurations.

Currently ORB, GLCM and k-means clustering features are available for extraction.
Read the skimage documentation to understand what settings are available.
"""

options = dict(
    images_directory = '/Users/tom/Masters-Project/Lichen-Images/Datasets/augmented-dataset/augmented-validation/Ordered',
    labels_path = '/Users/tom/Masters-Project/Lichen-Images/Datasets/augmented-dataset/augmented-validation/Ordered/labels.csv',
    extractor = 'image-processing',
    parser = 'csv',
    output_directory = '/Users/tom/Masters-Project/Lichen-Images/Datasets/testing_extraction/',
    file_name = 'features.csv',
    # The individual feature image processing feature extraction settings
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
