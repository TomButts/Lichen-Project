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
import getopt

from tools.evaluation.load import load
from tools.evaluation.metrics import metrics

def evaluate(directory_path, mode=None):
    if directory_path is None:
        print('\nA directory with model and training data is required')
        exit()

    items = load(directory_path)
    scores = None

    # if 'mlp' in items['config']:
    #     model_options = items['config']['mlp']
    # else:
    #     model_options = items['config']['svc']
    #
    # # regular classifications
    # metrics(items['model'], items['data'], scores, mode)

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
                # TODO change
                evaluate(arg)

                exit()
            elif opt in ('-u'):
                usage()
                exit()
    except getopt.GetoptError as err:
        # print(err)
        usage()
        exit()
