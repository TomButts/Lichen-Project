import sys
import os
import time
import pickle

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../utils')
sys.path.append(utils_path)

import yes_no_prompt

def export(classifiers, data, selectors, options, info, scaler=None, folder_name=None, output_directory=None):
    """Handles saving training objects

    Args:
        classifiers: classifier objects
        data: dictionary of training and testing feature data
        selectors: selector objects
        options: the configuration settings
        info: dataset meta info
        scaler: a fitted scaler object
        folder_name: the output folder name
        output_directory: directory to save output folder to
    """
    if folder_name == None:
        now = time.strftime("%d-%b-%H%M%S")
        folder_name = '/evaluation-' + now

    if output_directory == None:
        output_directory = os.path.abspath('output/evaluations/')

    results_directory = output_directory + folder_name

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    else:
        print('A training output of the same name exists.\n')
        replace_output = yes_no_prompt.yes_or_no('Replace Training Folder?')

        if not replace_output:
            exit()

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
    """Serialises and saves an object

    Args:
        item: object to serialise and save
        name: the output file name
        results_directory: the directory to save to
    """
    filename = results_directory + '/' + name + '.sav'
    pickle.dump(item, open(filename, 'wb'))
