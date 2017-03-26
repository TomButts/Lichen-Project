# get data from csv
# conditional preprocessing
# conditionally apply feature selection
# train test split
# shuffle
# return data
#
# Should include pre feature selection metrics?

import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import read_features_csv as features_csv

def get_data(features_path):

    # TODO: if read_features csv is not needed elsewhere just paste it in here
    targets, data = features_csv.read_features_csv(features_path)

    return data, targets
