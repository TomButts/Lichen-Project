import pickle
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris()

X = iris['data']
y = iris['target']

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()

# fit only to the training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

filename = 'mlp_saved_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

predictions = loaded_model.predict(X_test)

print('confusion matrix:\n')
print(confusion_matrix(y_test, predictions))
print('\nclassification report:\n')
print(classification_report(y_test, predictions))
