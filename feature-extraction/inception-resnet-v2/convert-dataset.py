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

targets = sio.loadmat(target_path)

image_dir = '/Users/tom/Masters-Project/102flowers'

features_csv = open('irv2_102flowers_features.csv', 'wb')

writer = csv.writer(features_csv, quoting = csv.QUOTE_ALL)

index = 0

length = len(targets['labels'][0])

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # print(os.path.join(image_dir, filename))
        # features = irft.inception_resnet_v2_features('file:///' + image_dir + '/' +filename)


        index += 1

        pb.print_progress_bar(index, length, prefix = 'Progress:', suffix = 'Complete')

        continue
    else:
        continue





# writer.writerow(targets['labels'])
