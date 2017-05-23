#!/usr/bin/env python
'''
Model Evalution:

Runs analysis on the trained model saved in /classification/output/mlp
or svc.

Args:
    directory_path: which folder to run on
    mode: leave blank for single folder

For usage details:
python evaluate.py -u
============================
'''

import sys
import os
import time
import getopt

from tools.evaluation.load import load
from tools.evaluation.reports import model, dataset, configuration

def evaluate(input_directory, output_directory=None):
    if input_directory is None:
        print('\nA directory with model and training data is required')
        exit()

    if output_directory is None:
        now = time.strftime("%d-%b-%H%M%S")
        output_directory = input_directory + '/evaluation-' + now

    if os.path.isdir(output_directory) == False:
        os.mkdir(output_directory)

    items = load(input_directory)

    if 'mlp' in items['config']:
        model_options = items['config']['mlp']
    else:
        model_options = items['config']['svc']

    # write dataset information csv
    dataset.write_dataset_info(items['info'], output_directory)

    grid_options = {'accuracy': items['accuracy_grid'], 'f1_macro': items['f1_macro_grid'], 'neg_log_loss': items['neg_log_loss_grid']}
    configuration.write_config_info(model_options, grid_options, items['config'], output_directory)

    models = {'accuracy': items['accuracy'], 'f1_macro': items['f1_macro'], 'neg_log_loss': items['neg_log_loss']}

    model.write_model_csv(
        items['best_parameters'],
        items['data'],
        models,
        output_directory
    )

def usage():
    print("\nEvaluation Tool\n")
    print("Arg1: must be a path to a directory containing saved training data\n")
    print("Options")
    print("-n: normal mode")
    print("-l: loop mode\n")
    print("Normal mode evaluates a single folder full of training data (config, data, model and info.sav). ")
    print("Results are printed to terminal\n")
    print("Loop mode loops through a folder of results folders and writes the evaluation data to a csv file.")


if __name__ == "__main__":
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'n:l:u:')

        for opt, arg in options:
            if opt in ('-n'):
                evaluate(arg)
                exit()
            elif opt in ('-u'):
                usage()
                exit()
    except getopt.GetoptError as err:
        # print(err)
        usage()
        exit()
