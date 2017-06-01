'''
The transform tool

The transform tool can work in rescaling mode or augmentation mode.

Using -a the tool creates slightly transformed copies of the images in a
folder. The input_directory is used as an argument.

Using -r the tool rescales a folder of images to a maximum size of 800.

Settings for each mode can be changed by modifying rescale.py or augment.py.
'''

import os, os.path
import sys
import getopt
from skimage.io import imsave
from progressbar import AdaptiveETA, ProgressBar, Percentage, Counter

def transform(directory_path, function, repeat=None):
    index = 0

    path, dirs, files = os.walk(directory_path).next()

    file_count = len(files)

    print('File count:\n' + str(file_count))
    print('\nRemember to update configs with transform factor!\n\n')

    output_directory = directory_path + '/output'
    output_directory = os.path.normpath(output_directory)

    if os.path.isdir(output_directory) == False:
        os.mkdir(output_directory)

    if repeat != None:
        # Make the progress bar accurate
        file_count = file_count * repeat

    widgets = [AdaptiveETA(), ' Completed: ', Percentage(), '  (', Counter(), ')']
    pbar = ProgressBar(widgets = widgets, max_value = file_count).start()

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png"):
            if repeat == None:
                image = function(directory_path + '/' + filename)
                transform_path = output_directory + '/' + function.__name__ + '-' + str(index) + '.jpg'
                imsave(transform_path, image)
                index += 1
                pbar.update(index)
            elif repeat > 1:
                for _ in range(repeat):
                    image = function(directory_path + '/' + filename)
                    transform_path = output_directory + '/' + function.__name__ + '-' + str(index) + '.jpg'
                    imsave(transform_path, image)
                    index += 1
                    pbar.update(index)
            continue
        else:
            continue

    pbar.finish()

def usage():
    print('\nApply an image transform to every image in a folder\n')
    print('-r: Rescale the images to fit a maximum size')
    print('-a: Create randomly rotated, scaled and mirrored images\n')
    print('arg1: The path to the folder')
    print('arg2: The number of transformed copies to make. Only available in -a mode.')


if __name__ == "__main__":
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'a:r:')
    except getopt.GetoptError as err:
         print(err)
         usage()
         exit()

    for opt, arg in options:
        if opt in ('-a'):
            from augment import augment

            # parse second argument as transform factor (int)
            try:
                int(remainder[0])
                tranform_factor = int(remainder[0])
            except:
                print('Second argument must be an integer.\n')
                exit()

            if tranform_factor <= 1:
                print('Transform factor must be greater than or equal to 1.\n')
                exit()

            transform(arg, augment, tranform_factor)
        elif opt in ('-r'):
            from rescale import adjust_size
            transform(arg, adjust_size)
