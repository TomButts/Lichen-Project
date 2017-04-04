from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

def probabilistic(classifier, data, options):
    scores = {}

    probabilities = classifier.predict_proba(data['X_test'])
    predictions = classifier.predict(data['X_test'])

    scores['log_loss'] = log_loss(data['y_test'], probabilities)
    print("\nLog loss: %0.3f\n" % scores['log_loss'])

    scores['neg_log_loss'] = cross_val_score(
        classifier, data['X'], data['y'], scoring="neg_log_loss")

    print(scores['neg_log_loss'])
    print(
        "\nNeg Log Losses:\n\nCrossVal1: %0.3f\nCrossVal2: %0.3f\nCrossVal3: %0.3f\n" %
        (scores['neg_log_loss'][0],
         scores['neg_log_loss'][1],
         scores['neg_log_loss'][2]))

    return scores
