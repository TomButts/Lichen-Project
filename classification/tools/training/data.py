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

def dataset_info(training_data, training_targets, validation_data, validation_targets):
    info = {'training': {}, 'validation': {}}

    info['training']['total'] = len(training_targets)
    info['training']['classes'] = len(set(training_targets))

    info['validation']['total'] = len(validation_targets)
    info['validation']['classes'] = len(set(validation_targets))

    info['class_names'] = ['Physcia', 'Xanthoria', 'Flavoparmelia', 'Evernia']

    training_class_count = {}
    validation_class_count = {}

    training_class_count = count_unique(training_targets, info['class_names'])

    info['training']['class_count'] = training_class_count

    validation_class_count = count_unique(validation_targets, info['class_names'])

    info['validation']['class_count'] = validation_class_count

    info['training']['features'] = len(training_data[0])
    info['validation']['features'] = len(validation_data[0])

    print(info)

    return info

def post_processing_info(info, training_data, training_targets, testing_targets, validation_data):
    info['training']['train'] = {}
    info['training']['test'] = {}

    info['training']['train']['total']= len(training_targets)
    info['training']['test']['total'] = len(testing_targets)

    info['training']['features_after_selection'] = len(training_data[0])
    info['validation']['features_after_selection'] = len(validation_data[0])

    training_class_count = {}
    testing_class_count = {}

    training_class_count = count_unique(training_targets, info['class_names'])

    info['training']['train'] = training_class_count

    testing_class_count = count_unique(testing_targets, info['class_names'])

    info['training']['test'] = testing_class_count

    print(info)

    return info

def count_unique(targets, class_names):
    class_count = {}

    for unique in set(targets):
        class_name = class_names[int(float(unique)) - 1]
        class_count[class_name] = targets.count(unique)

    return class_count
