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
from tools.evaluation.load import load
from tools.evaluation.reports.multi_class import multi_class
from tools.evaluation.reports.probabilistic import probabilistic

def evaluate(directory_path, save_path=None):
    if directory_path == None:
        print('\nA directory with model and training data is required')
        exit()

    items = load(directory_path)
    score = None

    if 'mlp' in items['config']:
        model_options = items['config']['mlp']
    else:
        model_options = items['config']['svc']

    if model_options['probability']:
        # probability model without calibration
        scores = probabilistic(items['model'], items['data'], items['config'])

    # regular classifications
    scores = multi_class(items['model'], items['data'], scores)

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
