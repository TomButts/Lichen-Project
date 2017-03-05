import cv2
import numpy as np

def grab_cut(imageToCut, processedImage=None):
    # optionally don't use a processed image to calculate grabcut mask
    if processedImage is None:
        processedImage = imageToCut

    mask = np.zeros(processedImage.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    height, width, channels = processedImage.shape

    # x,y,width,height
    rect = (1, 1, width, height)

    cv2.grabCut(
        processedImage,
        mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return imageToCut * mask2[:, :, np.newaxis]
