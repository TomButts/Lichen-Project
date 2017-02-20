from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt
import UtilityFunctions as utils
from Segmentation.Crop import Crop

img = cv2.imread("D:\Python\Lichen\images\lichen.png")
cv2.namedWindow('lichen')

clonedImage = img.copy()

# Use a copy so roi can be drawn then thrown away
cropWindow = Crop(clonedImage)

cv2.setMouseCallback('lichen', cropWindow.crop)

while True:
	# display the image and wait for a keypress
	cv2.imshow("lichen", cropWindow.imageToCrop)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		img = clonedImage.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

cv2.destroyAllWindows()

meanShiftedImage = cv2.pyrMeanShiftFiltering(img, 5, 60, 3)

median = cv2.medianBlur(meanShiftedImage, 5)

meanShiftedImage2 = cv2.pyrMeanShiftFiltering(median, 5, 60, 3)

hsv = cv2.cvtColor(meanShiftedImage2, cv2.COLOR_BGR2HSV)

image = hsv.reshape((hsv.shape[0] * hsv.shape[1], 3))

clt = KMeans(5)

clt.fit(image)

hist = utils.centroid_histogram(clt)
colours = utils.main_colours(hist, clt.cluster_centers_)

bar = utils.plot_colors(hist, clt.cluster_centers_)

bounds = utils.BGR2HSVColourBoundaries(colours[2], 15)

mask = cv2.inRange(hsv, bounds[0], bounds[1])
res = cv2.bitwise_and(hsv, hsv, mask = mask)

# cannyEdges = cv2.Canny(meanShiftedImage2, 100, 200)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
# closedEdges = cv2.morphologyEx(cannyEdges, cv2.MORPH_CLOSE, kernel)

# _, contours, _ = cv2.findContours(cannyEdges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
#
# for c in contours:
# 	peri = cv2.arcLength(c, True)
# 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
# 	cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

cv2.imshow('mask', mask)
cv2.waitKey()

cv2.imshow('res', res)
cv2.waitKey()

cv2.imshow('hsv', hsv)
cv2.waitKey()

cv2.imshow('bar', bar)
cv2.waitKey()

# plt.subplot(121),plt.imshow(meanShiftedImage2,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(cannyEdges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
