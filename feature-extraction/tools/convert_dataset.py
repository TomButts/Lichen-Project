import csv
import numpy as np
import sys
import os
import time
import imp
from progressbar import AdaptiveETA, ProgressBar, Percentage, Counter

from time import sleep

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import yes_no_prompt

# enable long console output
# np.set_printoptions(threshold = sys.maxint)

def convert_dataset(options, mode=None):
    """Runs a feature conversion function on a folder of images.

       The feature extractor, image directory, labels path, parser type and
       output path used in the feature extraction process are all defined by
       a configuration file. These configuration files can be found in:
       /feature-extraction/configs/

    Args:
        options: The dictionary in the configuration file named options.
    """
    # Load feature extractor
    extractors_base_dir = os.path.abspath('../feature-extraction/extractors/')
    extractor_path = extractors_base_dir + '/' + options['extractor'] + '/' +'feature_extractor.py'
    extractor = imp.load_source('feature_extractor', extractor_path)

    # Load parser
    parsers_base_dir = os.path.abspath('../feature-extraction/tools/parsers/')
    parser_path = parsers_base_dir + '/' + options['parser'] + '/' + 'parse.py'
    parser = imp.load_source('parse', parser_path)

    # parse labels file
    targets = parser.parse(options['labels_path'])
    targets = map(int, targets)

    # Handle creation of csv
    output_path = options['output_directory'] + '/' + options['file_name']

    if not os.path.exists(options['output_directory']):
        # if the output directory doesnt exist create it
        os.makedirs(options['output_directory'])

        # then create the features file and csv writer
        features_csv = open(output_path, 'wb')
        writer = csv.writer(features_csv, quoting = csv.QUOTE_ALL)
    else:
        # if the output path does exist ask to replace it
        print('A features file of the same name exists.\n')
        replace_output = yes_no_prompt.yes_or_no('Replace Features File?')

        if not replace_output:
            # exit if they do not wish to replace it
            exit()
        else:
            # overwrite the features file
            features_csv = open(output_path, 'wb')
            writer = csv.writer(features_csv, quoting = csv.QUOTE_ALL)

    # progress bar set up
    widgets = [AdaptiveETA(), ' Completed: ', Percentage(), '  (', Counter(), ')']
    pbar = ProgressBar(widgets = widgets, max_value = len(targets)).start()

    index = 0

    # loop through image files
    for filename in os.listdir(options['images_directory']):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            # get list of features
            image_path = 'file:///' + options['images_directory'] + '/' + filename

            features = extractor.feature_extractor(image_path, options)

            # prepend label to features list
            features = np.insert(features, 0, targets[index])

            # write row to csv
            writer.writerow(features)

            # If a large feature batch is ran the cpu may heat up to
            # the point where peformance is effected. If this is the case
            # use the function in intermitent sleep mode
            if mode == 's':
                if index % 25 == 0:
                    sleep(500)

            # update counter
            index += 1

            # update progress bar
            pbar.update(index)

            continue
        else:
            continue

    pbar.finish()
