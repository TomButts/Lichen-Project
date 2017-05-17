"""
This file handles functions that evaluate models trained using
the training system. The evaluations are outputted to csv files.

"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix
import pandas as pd
import csv

def write_model_csv(best_parameters, data, models, directory_path):
    model_parameters(best_parameters, directory_path)

    metrics(data['X_val'], data['y_val'], models, directory_path)

def model_parameters(best_parameters, directory_path):
    output_path = directory_path + '/model_parameters.csv'

    evaluation_csv = open(output_path, 'wb')
    writer = csv.writer(evaluation_csv, quoting=csv.QUOTE_ALL)

    tested_params = []
    all_params = []

    tested_param_headers = best_parameters['f1_macro']['test'].keys()
    tested_param_headers.insert(0, 'scoring_method')

    all_param_headers = best_parameters['f1_macro']['all'].keys()
    all_param_headers.insert(0, 'scoring_method')

    for scoring_method, params in best_parameters.iteritems():
        test = [scoring_method]

        for param in params['test'].iteritems():
            # pick the value of the tuple
            test.append(param[1])

        tested_params.append(test)

        all_p = [scoring_method]

        for param in params['all'].iteritems():
            # pick the value of the tuple
            all_p.append(param[1])

        all_params.append(all_p)

    writer.writerow(tested_param_headers)

    for row in tested_params:
        writer.writerow(row)

    # line break
    writer.writerow('')

    writer.writerow(all_param_headers)

    for row in all_params:
        writer.writerow(row)

def metrics(validation_data, validation_targets, models, directory_path):
    output_path = directory_path + '/model_parameters.csv'

    # append to the end of the csv created for model params
    evaluation_csv = open(output_path, 'a')
    writer = csv.writer(evaluation_csv, quoting=csv.QUOTE_ALL)
    writer.writerow('')

    score_headers = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Log Loss']
    writer.writerow(score_headers)

    confusion_matrices = {}

    for name, model in models.iteritems():
        predictions = model.predict(validation_data)

        # TODO add config check
        probabilities = model.predict_proba(validation_data)
        accuracy = accuracy_score(validation_targets, predictions)
        f1 = f1_score(validation_targets, predictions, average='weighted')
        precision = precision_score(validation_targets, predictions, average='weighted')
        recall = recall_score(validation_targets, predictions, average='weighted')
        confusion = confusion_matrix(validation_targets, predictions)
        loss = log_loss(validation_targets, probabilities)

        score_row = [name, accuracy, f1, precision, recall, loss]
        writer.writerow(score_row)

        confusion_matrices[name] = confusion

    writer.writerow('')

    for name, matrix in confusion_matrices.iteritems():
        writer.writerow([name])

        for row in matrix:
            writer.writerow(row)

        writer.writerow('')
