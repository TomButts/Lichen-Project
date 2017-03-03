import sys
import os

dir = os.path.dirname(__file__)
slimFolderPath = os.path.join(dir, '../../Dependencies/models/slim')

sys.path.append(slimFolderPath)

from datasets import dataset_utils
import tensorflow as tf

# Model checkpoint
url = "http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz"

# Relative download directory
checkpointsFolderPath = os.path.join(dir, '../classification/checkpoints')

if not tf.gfile.Exists(checkpointsFolderPath):
    tf.gfile.MakeDirs(checkpointsFolderPath)

dataset_utils.download_and_uncompress_tarball(url, checkpointsFolderPath)
