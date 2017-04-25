from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

def probabilistic(classifier, data, options, mode=None):
    scores = {}

    probabilities = classifier.predict_proba(data['X_val'])
    predictions = classifier.predict(data['X_val'])

    if mode == 'loop':
        scores['neg_log_loss'] = cross_val_score(
            classifier, data['X_val'], data['y_val'], cv=4, scoring="neg_log_loss")
    else:
        scores['log_loss'] = log_loss(data['y_val'], probabilities)
        print("\nLog loss: %0.3f" % scores['log_loss'])

        scores['neg_log_loss'] = cross_val_score(
            classifier, data['X_val'], data['y_val'], cv=4, scoring="neg_log_loss")

        print(
            "\nNeg Log Losses:\nCrossVal1: %0.4f\nCrossVal2: %0.4f\nCrossVal3: %0.4f\nCrossVal4: %0.4f\n" %
            (scores['neg_log_loss'][0],
             scores['neg_log_loss'][1],
             scores['neg_log_loss'][2],
             scores['neg_log_loss'][3]))

    return scores
