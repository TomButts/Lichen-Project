import csv
import numpy as np
import sys
import os
import time
from progressbar import AdaptiveETA, ProgressBar, Percentage, Counter

# enable long console output
# np.set_printoptions(threshold = sys.maxint)

def convert_dataset(directory_path, target_path, target_parser, feature_extractor, output_path):
    """Runs a feature conversion function on a folder of images

    Args:
        directory_path: Path to image folder
        target_path: The path to the labels file
        target_parser: A function to parse the labels file and return a list of labels
        feature_extractor: A function to extract features from an image and return a list of features_csv
        output_path: The path to the output file where class labels and features lists are combined in a csv
    """
    targets = target_parser(target_path)

    targets = map(int, targets)

    # create csv to write to
    features_csv = open(output_path, 'wb')

    # csv writer object
    writer = csv.writer(features_csv, quoting = csv.QUOTE_ALL)

    # progress bar set up
    widgets = [AdaptiveETA(), ' Completed: ', Percentage(), '  (', Counter(), ')']
    pbar = ProgressBar(widgets = widgets, max_value = len(targets)).start()

    # counter
    index = 0

    # loop through image files
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            # get list of features
            features = feature_extractor('file:///' + directory_path + '/' + filename)

            # prepend label to features list
            features = np.insert(features, 0, targets[index])

            # write row to csv
            writer.writerow(features)

            # update counter
            index += 1

            # update progress bar
            pbar.update(index)

            continue
        else:
            continue

    pbar.finish()
