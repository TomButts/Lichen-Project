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
from tools.evaluation.reports import metrics, dataset

def evaluate(input_directory, output_directory=None):
    if input_directory is None:
        print('\nA directory with model and training data is required')
        exit()

    if output_directory == None:
        now = time.strftime("%d-%b-%H%M%S")
        output_directory = '/Users/tom/Masters-Project/Lichen-Project/classification/output/evaluations/evaluation-' + now

    if os.path.isdir(output_directory) == False:
        os.mkdir(output_directory)

    items = load(input_directory)

    dataset.write_dataset_info(items['info'], output_directory)

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
