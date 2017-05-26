"""
Example configuration file

An example configuration file that will run the inception feature
extractor on an ordered images directory. The labels file in this case
is a csv type file.
"""

options = dict(
    images_directory = '/Users/tom/Masters-Project/Lichen-Images/Datasets/augmented-dataset/augmented-validation/Ordered',
    labels_path = '/Users/tom/Masters-Project/Lichen-Images/Datasets/augmented-dataset/augmented-validation/Ordered/labels.csv',
    extractor = 'inception-resnet-v2',
    parser = 'csv',
    output_directory = '/Users/tom/Masters-Project/Lichen-Images/Datasets/testing_extraction/',
    file_name = 'features.csv'
)
