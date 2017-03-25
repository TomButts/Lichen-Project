import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
import yes_no_prompt
import pickle
import csv
import read_features_csv as features_csv
import time

targets, data = features_csv.read_features_csv('/Users/tom/Masters-Project/Output-Files/lichen.csv')

# assign training dataset
X = data
y = targets

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
selector.transform(X)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#
# X_train = sel.fit_transform(X_train)
# X_test = sel.transform(X_test)

max_abs_scaler = MaxAbsScaler()

X_train = max_abs_scaler.fit_transform(X_train)
X_test = max_abs_scaler.transform(X_test)

svc = SVC(probability=False, kernel="rbf", C=2.8, gamma=.0073)

svc.fit(X_train, y_train)

predictions = svc.predict(X_test)

print('confusion matrix:\n')
print(confusion_matrix(y_test, predictions))
print('\nclassification report:\n')
print(classification_report(y_test, predictions))
