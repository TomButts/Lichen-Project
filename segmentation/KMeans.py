import cv2
import numpy as np

def cluster_colours(image, clusters):
    Z = image.reshape((image.shape[0] * image.shape[1], 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    K = clusters
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)

    res = center[label.flatten()]

    return res.reshape((image.shape))
