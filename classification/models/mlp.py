"""
MLP model selection function

"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def mlp(training_data, training_targets, options):
    mlp = {}

    print("Fitting MLP Classifier:\n")

    for score in options['scoring_strategies']:
        if score == 'neg_log_loss' and options['probability'] == False:
            print('Model must be trained in probability mode for neg_log_loss gridsearch strategy')
        else:
            classifier = GridSearchCV(MLPClassifier(alpha=0.01), options['tuned_parameters'], cv=5, scoring=score)

            classifier.fit(training_data, training_targets)

            mlp[score] = classifier

    return mlp
