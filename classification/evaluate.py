#!/usr/bin/env python
'''
Runs sklearn scoring and metric functions

Runs analysis on the trained model saved in /classification/output/mlp
or svc.

Args:
    directory_path: which folder to run on
    save_path: optional save path, defaults to /classification/output/evaluations directory
'''

import sys
from tools.evaluation.load import load

def evaluate(directory_path, save_path=None):
    if directory_path == None:
        print('\nA directory with model and training data is required')
        exit()

    items = load(directory_path)

    



if __name__ == "__main__":
    # get arguments
    print(__doc__)

    directory_path = None
    save_path = None

    if len(sys.argv) > 1:
        directory_path = sys.argv[1]

    if len(sys.argv) > 2:
        save_path = sys.argv[2]

    # Run evaluate
    evaluate(directory_path, save_path)
