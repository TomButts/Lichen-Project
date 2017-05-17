"""
This file handles the top level extraction process.

The image processing feature extractor is used for extraction.

"""

import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import convert_dataset
import parse_csv as target_parser
import time
import features

directory = os.path.abspath('/Users/tom/Masters-Project/Lichen-Images/Datasets/augmented-dataset/aug-seg-validation/Ordered')

targets_path = directory + '/labels.csv'

now = time.strftime("%Y%m%d-%H%M%S")

output_path = '/Users/tom/Masters-Project/Lichen-Images/Datasets/augmented-dataset/aug-seg-validation/validation-k-glcm-' + now + '.csv'

convert_dataset.convert_dataset(directory, targets_path, target_parser.parse, features.features, output_path)
