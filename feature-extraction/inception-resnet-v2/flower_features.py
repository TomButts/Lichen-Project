import sys
import os
import inception_features as incep_res

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import convert_dataset
import parse_flowers as target_parser
import time

# enable long console output
# np.set_printoptions(threshold = sys.maxint)

directory = os.path.abspath('../../../flowers')

targets_path = directory + '/imagelabels.mat'

now = time.strftime("%Y%m%d-%H%M%S")

output_path = 'output/flowers-' + now + '.csv'

convert_dataset.convert_dataset(directory, targets_path, target_parser.parse, incep_res.inception_resnet_v2_features, output_path)
