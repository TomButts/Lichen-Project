import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import yes_no_prompt
import pickle
import csv
import read_features_csv as features_csv
import time

targets, data = features_csv.read_features_csv('/Users/tom/Masters-Project/Output-Files/lichen.csv')

# assign training dataset
X = data
y = targets

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()

# fit only to the training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(20, 60, 30), max_iter=2500)

# train the network
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print('confusion matrix:\n')
print(confusion_matrix(y_test, predictions))
print('\nclassification report:\n')
print(classification_report(y_test, predictions))

save_model = yes_no_prompt.yes_or_no('Save model?')

now = time.strftime("%Y%m%d-%H%M%S")

if save_model:
    filename = 'checkpoints/mlp_saved_model_' + now + '.sav'
    pickle.dump(mlp, open(filename, 'wb'))
