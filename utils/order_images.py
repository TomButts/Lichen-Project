import os
import sys
import csv
import shutil

dir = os.path.dirname(__file__)
lichen_images = os.path.join(dir, '../../Lichen-Images/')
sys.path.append(lichen_images)

folders = ['Physcia', 'Xanthoria']

output_folder = 'Ordered'
labels_file = 'labels.csv'

labels_csv = open(lichen_images + labels_file, 'wb')

writer = csv.writer(labels_csv, quoting = csv.QUOTE_ALL)

labels = []

folder_number = 1
image_number = 1

for folder in folders:
    for filename in os.listdir(lichen_images + folder):
        if filename.endswith(".jpg") or filename.endswith(".NEF") or filename.endswith(".jpeg"):
            # rename image in output folder
            shutil.copy(lichen_images + folder + '/' + filename, lichen_images + output_folder + '/image_' + str(image_number).zfill(4) + '.jpg')

            # append to label array
            labels.append(folder_number)

            image_number += 1
    folder_number += 1

writer.writerow(labels)
