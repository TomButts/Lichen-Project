import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import read_features_csv as features_csv
import pickle
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

targets, data = features_csv.read_features_csv('../../feature-extraction/inception-resnet-v2/output/lichen.csv')

X = data
y = targets

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()

# fit only to the training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

filename = 'checkpoints/mlp_saved_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

predictions = loaded_model.predict(X_test)

print(y_test)
print('confusion matrix:\n')
print(confusion_matrix(y_test, predictions))
print('\nclassification report:\n')
print(classification_report(y_test, predictions))
