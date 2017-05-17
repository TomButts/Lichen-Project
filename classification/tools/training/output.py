"""
These functions handle serialising training objects and saving them
to an ouput folder.
"""

import os
import time
import pickle

def export(classifiers, data, selectors, options, info, scaler=None, folder_name=None):
    if 'mlp' in options:
        model_options = options['mlp']
        output_path = os.path.abspath('output/evaluations/Model/MLP/')
    else:
        model_options = options['svc']
        output_path = os.path.abspath('output/evaluations/Model/SVC/')

    # print(output_path)
    now = time.strftime("%d-%b-%H%M%S")

    score = classifiers['accuracy'].score(data['X_val'], data['y_val'])

    results_directory = output_path + '/' + now + '-' + format(score, '.2f')

    if folder_name != None:
        results_directory = output_path + '/' + folder_name

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    save(data, 'data', results_directory)

    best_parameters = {}

    for name, classifier in classifiers.iteritems():
        # Save the trained models
        save(classifier.best_estimator_, name, results_directory)

        # Save grid search config
        save(classifier.cv_results_, name + '_grid', results_directory)

        # Save parameters for results
        best_parameters[name] = {'test': classifier.best_params_, 'all': classifier.best_estimator_.get_params()}

    save(best_parameters, 'best_parameters', results_directory)

    save(selectors, 'selectors', results_directory)

    save(options, 'config', results_directory)

    save(info, 'info', results_directory)

    if scaler != None:
        save(scaler, 'scaler', results_directory)

    return results_directory

def save(item, name, results_directory):
    filename = results_directory + '/' + name + '.sav'
    pickle.dump(item, open(filename, 'wb'))
