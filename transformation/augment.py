import os, os.path
import transform as t
from skimage.io import imsave
from progressbar import AdaptiveETA, ProgressBar, Percentage, Counter

directory_path = os.path.abspath('../../Lichen-Images/Test')

index = 0

path, dirs, files = os.walk(directory_path).next()

file_count = len(files)

print('File count:\n' + file_count)
print('\nRemember to update configs with transform factor!\n\n')

widgets = [AdaptiveETA(), ' Completed: ', Percentage(), '  (', Counter(), ')']
pbar = ProgressBar(widgets = widgets, max_value = file_count).start()

for filename in os.listdir(directory_path):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        image = t.transform(directory_path + '/' + filename)

        transform_path = directory_path + '/transform-' + str(index) + '.jpg'

        imsave(transform_path, image)

        index += 1

        pbar.update(index)
        continue
    else:
        continue

pbar.finish()
