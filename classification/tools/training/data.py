import sys
import os
import csv

def get_data(features_path):
    """Read in data from a features csvfile

    Args:
        features_path: the full path to the features csvfile
    Returns:
        data: array containing features
        targets: a corresponding array of class labels
    """
    targets = []
    data = []

    with open(features_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '\"')

        for row in reader:
            # create target list
            targets.append(row[0])

            # create data list
            data.append(row[1:])

    return data, targets

def dataset_info(training_data, training_targets, validation_data, validation_targets):
    """Collects dataset meta info pre feature selection

    Args:
        training_data: the training features
        training_targets: training labels
        validation_data: feature data set aside for testing
        validation_targets: corresponding testing labels
    Returns:
        info: a dictionary containing meta information
    """
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

    return info

def post_processing_info(info, training_data, validation_data):
    """Appends feature meta info to the dataset meta infromation dictionary

    Args:
        info: a meta data dict made by dataset_info()
        training_data: training features
        validation_data: testing features
    Returns:
        info: augmented meta info dict
    """
    info['training']['features_after_selection'] = len(training_data[0])
    info['validation']['features_after_selection'] = len(validation_data[0])

    return info

def count_unique(targets, class_names):
    """Counts the number of different classes in a feature set

    Args:
        targets: the class labels
        class_names: array of class names in the same order as labels
        ie for labels [1, 2, 3] [cat, dog, boat] would link cat to 1, dog to 2
        and boat to 3
    """
    class_count = {}

    for unique in set(targets):
        class_name = class_names[int(float(unique)) - 1]
        class_count[class_name] = targets.count(unique)

    return class_count
