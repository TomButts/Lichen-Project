import scipy.io as sio
import csv
import numpy as np
import sys
import os
import inception_features as incep_res
import time
from progressbar import AdaptiveETA, ProgressBar, Percentage, Counter

# enable long console output
# np.set_printoptions(threshold = sys.maxint)

image_dir = os.path.abspath('../../../flowers')

targets_path = image_dir + '/imagelabels.mat'

targets = sio.loadmat(targets_path)
targets = targets['labels'][0]

# uncomment for parsing a csv list of target labels
# with open(targets_path, 'rb') as csvfile:
#     reader = csv.reader(csvfile, delimiter = ',', quotechar = '\"')
#
#     for row in reader:
#         targets = row

targets = map(int, targets)

# Create csv to write to
features_csv = open('output/flowers.csv', 'wb')

# csv writer object
writer = csv.writer(features_csv, quoting = csv.QUOTE_ALL)

# progress bar set up
widgets = [AdaptiveETA(), ' Completed: ', Percentage(), '  (', Counter(), ')']
pbar = ProgressBar(widgets = widgets, max_value = len(targets)).start()

# counter
index = 0

# loop through image files
for filename in os.listdir(image_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
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
