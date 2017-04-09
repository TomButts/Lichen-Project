#!/usr/bin/env python
'''
Model Evalution:

Runs analysis on the trained model saved in /classification/output/mlp
or svc.

Args:
    directory_path: which folder to run on
    save_path: optional save path, defaults to /classification/output/evaluations directory
'''

import sys
import os
import getopt

from tools.evaluation.load import load
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
        scores = probabilistic(items['model'], items['data'], items['config'])

    # regular classifications
    scores = multi_class(items['model'], items['data'], scores, mode)

    return scores

def usage():
    print("\nEvaluation Tool\n")
    print("Options")
    print("-n: normal mode")
    print("-l: loop mode\n")
    print("Normal mode evaluates a single folder full of training data (config, data, model and info.sav). ")
    print("Results are printed to terminal\n")
    print("Loop mode loops through a folder of results folders and writes the evaluation data to a csv file.")

if __name__ == "__main__":
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'n:l:')
    except getopt.GetoptError as err:
         print(err)
         usage()
         exit()

    for opt, arg in options:
        if opt in ('-n'):
            evaluate(arg)
        elif opt in ('-l'):
            csv_scores = []

            for subdir, dirs, _ in os.walk(arg):
                if subdir != arg:
                    csv_scores.append(evaluate(subdir, mode='loop'))

                    # TODO create a func in tools that sorts through this heap of shit
                    # and writes it nicely into a csv
