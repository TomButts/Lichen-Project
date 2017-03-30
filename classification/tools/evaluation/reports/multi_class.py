from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def multi_class(classifier, data, scores=None):
    # Get predictions
    predictions = classifier.predict(data['X_test'])

    if scores == None:
        scores = {}

    # Cross Validation Score
    scores['cross_validation_mean'] = cross_validation_scores = cross_val_score(classifier, data['X'], data['y'], cv=5)
    print("CVS Accuracy: %0.2f (+/- %0.2f)" % (cross_validation_scores.mean(), cross_validation_scores.std() * 2))

    # Accuracy score
    scores['accuracy_score'] = accuracy_score(data['y_test'], predictions, normalize=True)
    print("\nClassification Percentage: %0.2f\n" % (scores['accuracy_score'] * 100))

    # Classification Report
    target_names = ['Physcia', 'Xanthoria', 'Flavoparmelia']
    scores['classification_report'] = classification_report(data['y_test'], predictions, target_names=target_names)
    print(scores['classification_report'])

    # Confusion Matrix
    scores['confusion_matrix'] = confusion_matrix(predictions, data['y_test'])
    print("\nConfusion Matrix: \n")
    print(scores['confusion_matrix'])

    return scores
