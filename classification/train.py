#!/usr/bin/env python
'''
Train Model:

This file trains a model using configuration settings found in:

/classification/configs/

============================
'''

import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../utils')
sys.path.append(utils_path)

import yes_no_prompt

from configs import mlp_model as config
# from configs import svc_model as config

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

options = config.options

# Original Custom
# folder_name = 'original-custom-5'
# X, y = get_data(
#     '/Users/tom/Masters-Project/Lichen-Images/Feature Sets/original-custom-5/training-80.csv')
#
# X_val, y_val = get_data(
#     '/Users/tom/Masters-Project/Lichen-Images/Feature Sets/original-custom-5/validation-20.csv')

# Augmented Custom
folder_name = 'augmented-custom-5'
X, y = get_data(
    '/Users/tom/Masters-Project/Lichen-Images/Feature Sets/augmented-custom-5/training-80/training.csv')

X_val, y_val = get_data(
    '/Users/tom/Masters-Project/Lichen-Images/Feature Sets/augmented-custom-5/validation-20/validation.csv')

# # Original Inception
# folder_name = 'original-inception'
# X, y = get_data(
#     '/Users/tom/Masters-Project/Lichen-Images/Feature Sets/original-inception/training-80.csv')
#
# X_val, y_val = get_data(
#     '/Users/tom/Masters-Project/Lichen-Images/Feature Sets/original-inception/validation-20.csv')
# #
# # Augmented Inception
# folder_name = 'augmented-inception'
# X, y = get_data(
#     '/Users/tom/Masters-Project/Lichen-Images/Feature Sets/augmented-inception/training-80/training.csv')
#
# X_val, y_val = get_data(
#     '/Users/tom/Masters-Project/Lichen-Images/Feature Sets/augmented-inception/validation-20/validation.csv')


info = dataset_info(X, y, X_val, y_val)

selectors = None

if 'selectors' in options:
    variance_threshold_selector, percentile_selector = fit_selectors(X, y, options['selectors'])

    X = transform_features(X, variance_threshold_selector, percentile_selector)

    X_val = transform_features(X_val, variance_threshold_selector, percentile_selector)

    selectors = {'variance': variance_threshold_selector, 'percentile': percentile_selector}

if 'scaling' in options:
    X, scaler = scale(X, options['scaling'])

    if scaler != None:
        X_val = scaler.transform(X_val)

X, y = shuffle(X, y)
X_val, y_val = shuffle(X_val, y_val)

info = post_processing_info(info, X, X_val)

if 'mlp' in options:
    model_options = options['mlp']
    classifiers = mlp(X, y, model_options)
else:
    model_options = options['svc']
    classifiers = svc(X, y, model_options)

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
print(info)

save_model = yes_no_prompt.yes_or_no('Save model?')

if save_model:
    # preserve the split data for later tests
    data = {
        'X': X,
        'y': y,
        'X_val': X_val,
        'y_val': y_val,
    }

    results_directory = export(classifiers, data, selectors, options, info, scaler, folder_name)
