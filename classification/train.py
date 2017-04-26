import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../utils')
sys.path.append(utils_path)

import yes_no_prompt

from configs import mlp_model as config

from models.mlp import mlp
from models.svc import svc

from tools.training.data import get_data, dataset_info, post_processing_info
from tools.training.feature_selection import fit_selectors, transform_features
from tools.training.preprocessing import scale, prepare
from tools.training.output import export

from sklearn.metrics import classification_report, confusion_matrix, log_loss
from itertools import groupby

import numpy as np

options = config.options

# /Users/tom/Masters-Project/Lichen-Project/feature-extraction/custom/output/lichen-20170410-165620.csv
# /Users/tom/Masters-Project/Lichen-Images/Datasets/datatset-01-04-17/transformed-classes-2/dataset-01-04-17.csv
#
# /Users/tom/Masters-Project/Lichen-Images/Datasets/datatset-01-04-17/transformed-classes-2/Split-Dataset/train-0.7-val-0.3
# /Users/tom/Masters-Project/Lichen-Images/Datasets/datatset-01-04-17/transformed-classes-2/Split-Dataset/train-0.7-validation-0.3/validation.csv

X, y = get_data(
    '/Users/tom/Masters-Project/Lichen-Images/Datasets/datatset-01-04-17/transformed-classes-2/dataset-01-04-17.csv')

X_val, y_val = get_data(
    '/Users/tom/Masters-Project/Lichen-Images/Datasets/datatset-01-04-17/transformed-classes-2/dataset-01-04-17.csv')

info = dataset_info(X, y, X_val, y_val)

selectors = None

if 'selectors' in options:
    variance_threshold_selector, percentile_selector = fit_selectors(X, y, options['selectors'])

    X = transform_features(X, variance_threshold_selector, percentile_selector)

    X_val = transform_features(X_val, variance_threshold_selector, percentile_selector)

    selectors = {'variance': variance_threshold_selector, 'percentile': percentile_selector}

if 'scaling' in options:
    X = scale(X, options['scaling'])
    X_val = scale(X_val, options['scaling'])

X_train, X_test, y_train, y_test = prepare(X, y)

info = post_processing_info(info, X_train, y_train, y_test, X_val)

if 'mlp' in options:
    model_options = options['mlp']
    classifiers = mlp(X_train, y_train, model_options)
else:
    model_options = options['svc']
    classifiers = svc(X_train, y_train, model_options)

# Print some intial analysis
if model_options['probability']:
    probabilities = classifiers['neg_log_loss'].predict_proba(X_val)

    print('Log Loss')
    print(log_loss(y_val, probabilities))

predictions = classifiers['accuracy'].predict(X_val)

print('\nconfusion matrix:')
print(confusion_matrix(y_val, predictions))

print('\nclassification report:\n')
print(classification_report(y_val, predictions))

print('\ndata info:')
print(info)

save_model = yes_no_prompt.yes_or_no('Save model?')

if save_model:
    # preserve the split data for later tests
    data = {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_val': X_val,
        'y_val': y_val,
    }

    results_directory = export(classifiers, data, selectors, options, info)
