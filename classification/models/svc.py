"""
SVC Model Selection Function
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def svc(training_data, training_targets, options):
    svc = {}

    print("Fitting SVC Classifier:\n")

    for score in options['scoring_strategies']:
        if score == 'neg_log_loss' and options['probability'] == False:
            print('Model must be trained in probability mode for neg_log_loss gridsearch strategy')
        else:
            classifier = GridSearchCV(SVC(C=1), options['tuned_parameters'], cv=5, scoring=score)

            classifier.fit(training_data, training_targets)

            svc[score] = classifier

    return svc
