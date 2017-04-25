import os
import time
import pickle


def export(classifier, data, selectors, options, info, calibrated_clf=None):
    if classifier.__class__.__name__ == 'MLPClassifier':
        output_path = os.path.abspath('output/mlp')
    else:
        output_path = os.path.abspath('output/svc/c-value/CNN')

    # print(output_path)
    now = time.strftime("%d-%b-%H%M%S")

    score = classifier.score(data['X_test'], data['y_test'])

    results_directory = output_path + '/' + now + '-' + format(score, '.2f')

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    save(data, 'data', results_directory)

    save(classifier, 'model', results_directory)

    save(selectors, 'selectors', results_directory)

    save(options, 'config', results_directory)

    save(info, 'info', results_directory)

    if calibrated_clf is not None:
        save(calibrated_clf, 'calib', results_directory)

    return results_directory


def save(item, name, results_directory):
    filename = results_directory + '/' + name + '.sav'
    pickle.dump(item, open(filename, 'wb'))
