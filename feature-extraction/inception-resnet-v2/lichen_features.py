import sys
import os
import inception_features as incep_res

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import convert_dataset
import parse_csv as target_parser
import time

directory = '/Users/tom/Masters-Project/Lichen-Images/Datasets/datatset-01-04-17/transformed-classes-2/Xanthoria-Ordered'

targets_path = directory + '/labels.csv'

now = time.strftime("%Y%m%d-%H%M%S")

output_path = '/Users/tom/Masters-Project/Lichen-Images/Datasets/datatset-01-04-17/transformed-classes-2/Xanthoria-' + now + '.csv'

print('\nSaving to:\n' + output_path)

convert_dataset.convert_dataset(directory, targets_path, target_parser.parse, incep_res.inception_resnet_v2_features, output_path)
