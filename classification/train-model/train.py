# call get data
# condtionally instantiate model from config
# train
# y or no to create output and save model
# create unique folder
# save model
# copy config.py
# create text file with predictions confusion matrix etc

import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)


import yes_no_prompt
from data import get_data
from models.mlp import mlp
from models.svc import svc
from feature_selection import select_features
from preprocessing import scale, prepare
from output import export
from configs import example as config
from sklearn.metrics import classification_report, confusion_matrix
from itertools import groupby

options = config.options

info = {}

X, y = get_data('/Users/tom/Masters-Project/Output-Files/lichen.csv')

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
    clf = mlp(X_train, y_train, options['mlp'])
else:
    clf = svc(X_train, y_train, options['svc'])

# print prediction info to determine value of model
predictions = clf.predict(X_test)

score = clf.score(X_test, y_test)

print('confusion matrix:\n')
print(confusion_matrix(y_test, predictions))
print('\nclassification report:\n')
print(classification_report(y_test, predictions))

print(info)
save_model = yes_no_prompt.yes_or_no('Save model?')

if save_model:
    export(clf, score, options, info)
