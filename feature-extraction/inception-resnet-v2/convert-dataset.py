from __future__ import print_function
import scipy.io as sio
import csv
import numpy as np
import sys
import os
import inception_features as incep_res

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import time
from progressbar import AdaptiveETA, ProgressBar, Percentage, Counter

np.set_printoptions(threshold = sys.maxint)

target_path = '/Users/tom/Masters-Project/imagelabels.mat'
image_dir = '/Users/tom/Masters-Project/flowers'

# read mat file with class info into array
targets = sio.loadmat(target_path)
targets = targets['labels'][0]

for key, value in enumerate(targets):
    if value == 74:
        targets = targets[0:key]
        break

# length and counter for progress bar calcs
length = len(targets)
index = 0

# Create csv to write to
features_csv = open('irv2_5flowers_features.csv', 'wb')

# csv writer object
writer = csv.writer(features_csv, quoting = csv.QUOTE_ALL)

# intialise headers array
headers = []

# write csv headers outside the loop
# Format: 'lables', 'feature_0001', ... , 'feature_n'
headers.append('label')

# extact features from image to get number of features
for filename in os.listdir(image_dir):
    feature_index = incep_res.inception_resnet_v2_features('file:///' + image_dir + '/' + filename)
    break

# add feature columns
for key in range(len(feature_index)):
    headers.append('feature_' + str(key))

writer.writerow(headers)

# progress bar set up
widgets = [AdaptiveETA(), ' Completed: ', Percentage(), '  (', Counter(), ')']
pbar = ProgressBar(widgets = widgets, max_value = length).start()

# loop through image files
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # get list of features
        features = incep_res.inception_resnet_v2_features('file:///' + image_dir + '/' + filename)

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
