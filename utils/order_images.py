'''
Order Images

Each image must be linked with class label information to convert a set of
images into features that can be used in supervised learning classification.

This script allows a group of folders containing images of the same type to be
ordered in one folder. The script also writes a csv labels file that corresponds
to the image order. Feature extractors in this project can be ran on the outputted
ordered images folder.

The input_directory, output_directory and the names of the folders within the
input_directory that should be included in the ordering process are found in:
/utils/order_config.py
'''

import os
import sys
import csv
import shutil

# import ordering settings
from order_config import options

dir = os.path.dirname(__file__)
lichen_images = os.path.join(dir, options['input_directory'])
sys.path.append(lichen_images)

base_directory = os.path.abspath(lichen_images)

if os.path.isdir(options['output_directory']) == False:
    os.mkdir(options['output_directory'])

labels_file = options['output_directory'] + '/labels.csv'

labels_csv = open(labels_file, 'wb')

writer = csv.writer(labels_csv, quoting = csv.QUOTE_ALL)

labels = []

folder_number = 1
image_number = 1

for folder in options['folder_names']:
    for filename in os.listdir(lichen_images + folder):
        if filename.endswith(".jpg") or filename.endswith(".NEF") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
            # rename image in output folder
            shutil.copy(base_directory + '/' + folder + '/' + filename, options['output_directory'] + '/image_' + str(image_number).zfill(4) + '.jpg')

            # append to label array
            labels.append(folder_number)

            image_number += 1
    folder_number += 1

writer.writerow(labels)
