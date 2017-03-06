import scipy.io as sio
import csv
import numpy as np
import sys
import os
import inception_features as irft

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import progress_bar as pb

np.set_printoptions(threshold = sys.maxint)

target_path = '/Users/tom/Masters-Project/imagelabels.mat'
image_dir = '/Users/tom/Masters-Project/102flowers'

# read mat file with class info into array
targets = sio.loadmat(target_path)
targets = targets['labels'][0]

# length and counter for progress bar calcs
length = len(targets)
index = 0

# Create csv to write to
features_csv = open('irv2_102flowers_features.csv', 'wb')

# csv writer object
writer = csv.writer(features_csv, quoting = csv.QUOTE_ALL)

# intialise headers array
headers = []

# loop through image files
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # extract features from image file
        features = irft.inception_resnet_v2_features('file:///' + image_dir + '/' +filename)

        if index == 0:
            # write csv headers
            # Format: 'lables', 'feature_0001', ... , 'feature_n'
            headers.append('label')

            # add feature columns
            for key, value in enumerate(features):
                headers.append('feature_' + str(key).zfill(4))

            writer.writerow(headers)

        # clear row array every iteration to avoid large array in memory
        row = []

        # add class label number to row array
        row.append(targets[index])

        # add features to row array
        for feature in features:
            row.append(feature)

        writer.writerow(row)

        index += 1

        # sexy progress bar
        pb.print_progress_bar(index, length, prefix = 'Progress:', suffix = 'Complete')
        continue
    else:
        continue
