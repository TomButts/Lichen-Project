# call get data
# condtionally instantiate model from config
# train
# y or no to create output and save model
# create unique folder
# save model
# copy config.py
# create text file with predictions confusion matrix etc
from models.mlp import mlp
from models.svc import svc
from get_data import get_data
from configs import example as options
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = get_data('/Users/tom/Masters-Project/Output-Files/lichen.csv', options)

if options.mlp:
    clf = mlp(X_train, y_train, options.mlp)
else:
    clf = svc(X_train, y_train, options.svc)

# print prediction info to determine value of model
predictions = clf.predict(X_test)

print('confusion matrix:\n')
print(confusion_matrix(y_test, predictions))
print('\nclassification report:\n')
print(classification_report(y_test, predictions))
