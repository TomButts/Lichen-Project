import sys
import os
import inception_features as incep_res

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import convert_dataset
import parse_csv as target_parser
import time

# enable long console output
# np.set_printoptions(threshold = sys.maxint)

directory = '/Users/tom/Masters-Project/Lichen-Images/Ordered'

targets_path = directory + '/labels.csv'

now = time.strftime("%Y%m%d-%H%M%S")

output_path = '/Users/tom/Masters-Project/Lichen-Project/feature-extraction/inception-resnet-v2/output/lichen-' + now + '.csv'

convert_dataset.convert_dataset(directory, targets_path, target_parser.parse, incep_res.inception_resnet_v2_features, output_path)
