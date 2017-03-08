import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import yes_no_prompt
import pickle
import csv

# temp test data
# iris = datasets.load_iris()
# X = iris['data']
# y = iris['target']

row_number = 0
targets = []
data = []

with open('../../feature-extraction/inception-resnet-v2/irv2_5flowers_features.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',', quotechar = '\"')

    for row in reader:
        if row_number == 0:
            # skip headers
            row_number += 1
        else:
            # create target list
            targets.append(row[0])

            # create data list
            data.append(row[1:])

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

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=1500)

# train the network
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print('confusion matrix:\n')
print(confusion_matrix(y_test, predictions))
print('\nclassification report:\n')
print(classification_report(y_test, predictions))

save_model = yes_no_prompt.yes_or_no('Save model?')

if save_model:
    filename = 'mlp_saved_model.sav'
    pickle.dump(mlp, open(filename, 'wb'))
