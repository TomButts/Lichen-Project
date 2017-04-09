from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def multi_class(classifier, data, scores=None, mode=None):
    # Get predictions
    predictions = classifier.predict(data['X_test'])

    if scores == None:
        scores = {}

    if mode == 'loop':
        scores['cross_val_f1'] = cross_val_score(classifier, data['X'], data['y'], cv=4, scoring="f1_weighted")
        scores['cross_val_precision'] = cross_val_score(classifier, data['X'], data['y'], cv=4, scoring="precision_weighted")
        scores['cross_val_recall'] = cross_val_score(classifier, data['X'], data['y'], cv=4, scoring="recall_weighted")

    # Cross Validation Score
    scores['cross_val_accuracy'] = cross_val_score(classifier, data['X'], data['y'], cv=4)
    print("CVS Accuracy: %0.2f (+/- %0.2f)" % (scores['cross_val_accuracy'].mean(), scores['cross_val_accuracy'].std() * 2))

    # Accuracy score
    scores['accuracy_score'] = accuracy_score(data['y_test'], predictions, normalize=True)
    print("\nClassification Percentage: %0.2f\n" % (scores['accuracy_score'] * 100))

    # Classification Report
    target_names = ['Physcia', 'Xanthoria', 'Flavoparmelia', 'Evernia']
    scores['classification_report'] = classification_report(data['y_test'], predictions, target_names=target_names)
    print(scores['classification_report'])

    # Confusion Matrix
    scores['confusion_matrix'] = confusion_matrix(predictions, data['y_test'])
    print("\nConfusion Matrix: \n")
    print(scores['confusion_matrix'])

    return scores
