"""
This file handles inception_resnet_v2 feature extraction.

"""

import numpy as np
import os
import sys

dir = os.path.dirname(__file__)
slimFolderPath = os.path.join(dir, '../../../Dependencies/slim')

sys.path.append(slimFolderPath)

import tensorflow as tf
import tensorflow.contrib.slim as slim
import urllib2

# from datasets import imagenet
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing

def inception_resnet_v2_features(image_path):
    # size of images inc-resnet is compatible with
    image_size = inception_resnet_v2.inception_resnet_v2.default_image_size

    checkpoint_path = os.path.join(dir, 'checkpoints/inception_resnet_v2_2016_08_30.ckpt')

    image_string = urllib2.urlopen(image_path).read()

    # JPEG format converted to unit8 tensor
    image = tf.image.decode_jpeg(image_string, channels=3)

    # inception specific pre processing
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)

    # the model accepts images in batches
    processed_images  = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2.inception_resnet_v2(processed_images,
                               num_classes=1001,
                               is_training=False)

    weights_from_file = slim.get_variables_to_restore(exclude=['logits'])

    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, weights_from_file)

    # last layer before fully connected layer
    features = end_points['PreLogitsFlatten']

    # run the image through the network
    with tf.Session() as sess:
        init_fn(sess)

        features = sess.run([features])

    return features[0][0]
