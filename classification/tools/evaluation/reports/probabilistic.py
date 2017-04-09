from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

def probabilistic(classifier, data, options):
    scores = {}

    probabilities = classifier.predict_proba(data['X_test'])
    predictions = classifier.predict(data['X_test'])

    scores['log_loss'] = log_loss(data['y_test'], probabilities)
    print("\nLog loss: %0.3f" % scores['log_loss'])

    scores['neg_log_loss'] = cross_val_score(
        classifier, data['X'], data['y'], cv=4, scoring="neg_log_loss")

    print(
        "\nNeg Log Losses:\nCrossVal1: %0.4f\nCrossVal2: %0.4f\nCrossVal3: %0.4f\nCrossVal4: %0.4f\n" %
        (scores['neg_log_loss'][0],
         scores['neg_log_loss'][1],
         scores['neg_log_loss'][2],
         scores['neg_log_loss'][3]))

    return scores
