#!/usr/bin/env python
'''
Model Evalution:

Runs analysis on the trained model saved in /classification/output/mlp
or svc.

Args:
    directory_path: which folder to run on
    mode: leave blank for single folder

Command line tool:
Can be used in normal -n and loop -l mode.
Argument is directory_path to training data

For usage details:
python evaluate.py -u
============================
'''

import sys
import os
import getopt

from tools.evaluation.load import load
from tools.evaluation.write import write_scores_csv
from tools.evaluation.reports.multi_class import multi_class
from tools.evaluation.reports.probabilistic import probabilistic

def evaluate(directory_path, mode=None):
    if directory_path == None:
        print('\nA directory with model and training data is required')
        exit()

    items = load(directory_path)
    scores = None

    if 'mlp' in items['config']:
        model_options = items['config']['mlp']
    else:
        model_options = items['config']['svc']

    if model_options['probability']:
        # probability model without calibration
        scores = probabilistic(items['model'], items['data'], items['config'], mode)

    # regular classifications
    scores = multi_class(items['model'], items['data'], scores, mode)

    return scores

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
            elif opt in ('-l'):
                csv_scores = []

                for subdir, dirs, _ in os.walk(arg):
                    if subdir != arg:
                        csv_scores.append(evaluate(subdir, mode='loop'))

                write_scores_csv(csv_scores)
                exit()
    except getopt.GetoptError as err:
         # print(err)
         usage()
         exit()
