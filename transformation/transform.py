import os, os.path
import sys
import getopt
from skimage.io import imsave
from progressbar import AdaptiveETA, ProgressBar, Percentage, Counter

def transform(directory_path, function):
    index = 0

    path, dirs, files = os.walk(directory_path).next()

    file_count = len(files)

    print('File count:\n' + str(file_count))
    print('\nRemember to update configs with transform factor!\n\n')

    output_directory = directory_path + '/output'
    output_directory = os.path.normpath(output_directory)

    if os.path.isdir(output_directory) == False:
        os.mkdir(output_directory)

    widgets = [AdaptiveETA(), ' Completed: ', Percentage(), '  (', Counter(), ')']
    pbar = ProgressBar(widgets = widgets, max_value = file_count).start()

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".JPG"):
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
            transform(arg, augment)
        elif opt in ('-r'):
            from rescale import adjust_size
            transform(arg, adjust_size)
