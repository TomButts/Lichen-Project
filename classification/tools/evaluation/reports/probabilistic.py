from sklearn.metrics import roc_curve, auc, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

def probabilistic(classifier, data, options):
    scores = {}

    probabilities = classifier.predict_proba(data['X_test'])
    predictions = classifier.predict(data['X_test'])

    scores['log_loss'] = log_loss(data['y_test'], probabilities)
    print("\nLog loss: %0.3f\n" % scores['log_loss'])

    scores['neg_log_loss'] = cross_val_score(
        classifier, data['X'], data['y'], scoring="neg_log_loss")
    print(
        "\nNeg Log Losses:\n\nPyschia: %0.3f\nXanthoria: %0.3f\nFlavoparmelia: %0.3f\n" %
        (scores['neg_log_loss'][0],
         scores['neg_log_loss'][1],
         scores['neg_log_loss'][2]))

    # TODO check config and do calibration metrics

    return scores
