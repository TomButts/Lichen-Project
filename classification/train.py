#!/usr/bin/env python
'''
The training tool

Runs model selection, feature selection and scaling on a feature set.
All relevant information and training objects are stored in pickle files.
Settings files that can be used for training are found in:
/classification/configs/

Run in -c mode and specify a config as an argument.
This will run regular training.

Run in -d mode to run training with a default unique output path.

Run -u for usage information.
'''

import sys
import os
import getopt
import imp
import warnings

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../utils')
sys.path.append(utils_path)

import yes_no_prompt

from models.mlp import mlp
from models.svc import svc

from tools.training.data import get_data, dataset_info, post_processing_info
from tools.training.feature_selection import fit_selectors, transform_features
from tools.training.preprocessing import scale
from tools.training.output import export

from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.utils import shuffle
from itertools import groupby

import numpy as np

def train(options):
    required_keys = (
        'training_path',
        'test_path',
        'output_directory',
        'folder_name')

    if not all(elem in options for elem in required_keys):
        print('Error: Missing input information.\n')
        exit()

    X, y = get_data(options['training_path'])

    X_test, y_test = get_data(options['test_path'])

    info = dataset_info(X, y, X_test, y_test)

    selectors = None

    if 'selectors' in options:
        variance_threshold_selector, percentile_selector = fit_selectors(
            X, y, options['selectors'])

        X = transform_features(
            X,
            variance_threshold_selector,
            percentile_selector)

        X_test = transform_features(
            X_test,
            variance_threshold_selector,
            percentile_selector)

        selectors = {
            'variance': variance_threshold_selector,
            'percentile': percentile_selector}

    if 'scaling' in options:
        X, scaler = scale(X, options['scaling'])

        if scaler is not None:
            X_test = scaler.transform(X_test)

    X, y = shuffle(X, y)
    X_test, y_test = shuffle(X_test, y_test)

    info = post_processing_info(info, X, X_test)

    if 'mlp' in options:
        model_options = options['mlp']
        classifiers = mlp(X, y, model_options)
    else:
        model_options = options['svc']
        classifiers = svc(X, y, model_options)

    # Print some intial analysis
    if model_options['probability']:
        probabilities = classifiers['neg_log_loss'].predict_proba(X_test)

        print('Log Loss')
        print(log_loss(y_test, probabilities))

    predictions = classifiers['accuracy'].predict(X_test)

    print('\nconfusion matrix:')
    print(confusion_matrix(y_test, predictions))
    print('\nclassification report:\n')
    print(classification_report(y_test, predictions))
    print(info)

    save_model = yes_no_prompt.yes_or_no('Save model?')

    if save_model:
        # preserve the split data for later tests
        data = {
            'X': X,
            'y': y,
            'X_test': X_test,
            'y_test': y_test,
        }

        results_directory = export(
            classifiers,
            data,
            selectors,
            options,
            info,
            scaler,
            options['folder_name'],
            options['output_directory'])

def usage():
    print("\nModel Training Tool\n")
    print("Call this training file with the -c option and the name of a training config file")
    print("An example config can be found in /classification/configs/")
    print("Do not include .py in the config name\n")
    print("Using the -d option sends the training output to a default directory:")
    print("/classification/output/evaluations/")

def warn(*args, **kwargs):
    pass

if __name__ == "__main__":
    warnings.warn = warn

    try:
        opts, remainder = getopt.getopt(sys.argv[1:], 'c:d:u:')

        for opt, arg in opts:
            if opt in ('-c'):
                config_base_dir = os.path.abspath('configs/')
                path = config_base_dir + '/' + arg + '.py'

                # 'arg' is the name of the config file without the file ext
                config = imp.load_source(arg, path)

                train(config.options)
                exit()
            elif opt in ('-d'):
                config_base_dir = os.path.abspath('configs/')
                path = config_base_dir + '/' + arg + '.py'

                # 'arg' is the name of the config file without the file ext
                config = imp.load_source(arg, path)

                config.options['output_directory'] = None
                config.options['folder_name'] = None

                train(config.options)
                exit()
            elif opt in ('-u'):
                usage()
                exit()
    except getopt.GetoptError as err:
        # print(err)
        usage()
        exit()
