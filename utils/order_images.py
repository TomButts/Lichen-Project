import os
import sys
import csv
import shutil

dir = os.path.dirname(__file__)
lichen_images = os.path.join(dir, '../../Lichen-Images/Datasets/datatset-01-04-17/seg-tr-classes/')
sys.path.append(lichen_images)

base_directory = os.path.abspath(lichen_images)

folders = ['Physcia', 'Xanthoria', 'Flavoparmelia', 'Evernia']

output_directory = base_directory + '/Ordered'

if os.path.isdir(output_directory) == False:
    os.mkdir(output_directory)

labels_file = output_directory + '/labels.csv'

labels_csv = open(labels_file, 'wb')

writer = csv.writer(labels_csv, quoting = csv.QUOTE_ALL)

labels = []

folder_number = 1
image_number = 1

for folder in folders:
    for filename in os.listdir(lichen_images + folder):
        if filename.endswith(".jpg") or filename.endswith(".NEF") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
            # rename image in output folder
            shutil.copy(base_directory + '/' + folder + '/' + filename, output_directory + '/image_' + str(image_number).zfill(4) + '.jpg')

            # append to label array
            labels.append(folder_number)

            image_number += 1
    folder_number += 1

writer.writerow(labels)
