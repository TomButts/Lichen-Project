from matplotlib import pyplot as plt

import numpy as np
import os
import sys

dir = os.path.dirname(__file__)
slimFolderPath = os.path.join(dir, '../../../Dependencies/models/slim')

sys.path.append(slimFolderPath)

import tensorflow as tf
import urllib2

from datasets import imagenet
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing


def inception_resnet_v2_features():
    # relative directory to checkpoints
    checkpointsFolderPath = os.path.join(dir, 'checkpoints/')

    slim = tf.contrib.slim

    # size of images inc-resnet is compatible with
    imageSize = inception_resnet_v2.inception_resnet_v2.default_image_size

    checkpointPath = os.path.join(dir, 'checkpoints/inception_resnet_v2_2016_08_30.ckpt')

    imagePath = 'file:///Users/tom/Masters-Project/Lichen-Project/images/parmelia.jpg'
    imageString = urllib2.urlopen(imagePath).read()

    # JPEG format converted to unit8 tensor
    image = tf.image.decode_jpeg(imageString, channels=3)

    # inception specific pre processing
    processedImage = inception_preprocessing.preprocess_image(image, imageSize, imageSize, is_training=False)

    # the model accepts images in batches
    processedImages  = tf.expand_dims(processedImage, 0)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2.inception_resnet_v2(processedImages,
                               num_classes=1001,
                               is_training=False)

    weightsRestoredFromFile = slim.get_variables_to_restore(exclude=['logits'])

    init_fn = slim.assign_from_checkpoint_fn(checkpointPath, weightsRestoredFromFile)

    features = end_points['PreLogitsFlatten']

    np.set_printoptions(threshold=sys.maxint)

    with tf.Session() as sess:
        init_fn(sess)

        features = sess.run([features])

        print len(features)
        print(features)

    return features
