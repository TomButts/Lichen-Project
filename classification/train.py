import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../utils')
sys.path.append(utils_path)

import yes_no_prompt

from configs import example as config

from models.mlp import mlp
from models.svc import svc

from tools.training.data import get_data
from tools.training.feature_selection import select_features
from tools.training.preprocessing import scale, prepare
from tools.training.output import export
from tools.training.calibrate import calibrate

from sklearn.metrics import classification_report, confusion_matrix, log_loss
from itertools import groupby

options = config.options

info = {}

X, y = get_data('/Users/tom/Masters-Project/Lichen-Images/Datasets/datatset-01-04-17/transformed-classes-2/dataset-01-04-17.csv')

info['total'] = len(y)
info['classes'] = len(set(y))

count = {}

for unique in set(y):
    count[unique] = y.count(unique)

info['count'] = count

# TODO check these operations are passing back the transformed shit
if 'selectors' in options:
    X, y = select_features(X, y, options['selectors'])

if 'scaling' in options:
    X = scale(X, options['scaling'])

X_train, X_test, y_train, y_test = prepare(X, y)

info['training'] = len(y_train)
info['test'] = len(y_test)

if 'mlp' in options:
    model_options = options['mlp']
    clf = mlp(X_train, y_train, options['mlp'])
else:
    model_options = options['svc']
    clf = svc(X_train, y_train, options['svc'])

calib = None

if 'calibration' in options:
    calib = calibrate(clf, X_train, y_train, options['calibration'])

# Print some intial analysis to determine
# if output should be saved
if model_options['probability']:
    probabilities = clf.predict_proba(X_test)

    print('Log Loss')
    print(log_loss(y_test, probabilities))

predictions = clf.predict(X_test)

print('\nconfusion matrix:')
print(confusion_matrix(y_test, predictions))
print('\nclassification report:\n')
print(classification_report(y_test, predictions))
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
    }

    results_directory = export(clf, data, options, info, calib)
