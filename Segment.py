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

roi = cv2.setMouseCallback('lichen', cropWindow.crop)

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

# coords are y1,y2, x1,x2 format
roi = img[cropWindow.refPt[0][0]:cropWindow.refPt[1][0], cropWindow.refPt[0][1]:cropWindow.refPt[1][1]]

# clone roi to compare
clonedRoi = roi.copy()

# meanShiftedImage = cv2.pyrMeanShiftFiltering(roi, 5, 60, 3)
#
# median = cv2.medianBlur(meanShiftedImage, 5)
#
# meanShiftedImage2 = cv2.pyrMeanShiftFiltering(median, 5, 60, 3)
#
# hsv = cv2.cvtColor(meanShiftedImage2, cv2.COLOR_BGR2HSV)

roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

#filters
roi = cv2.pyrMeanShiftFiltering(roi, 15, 60, 3)
roi = cv2.medianBlur(roi, 5)
roi = cv2.pyrMeanShiftFiltering(roi, 5, 60, 3)

Z = roi.reshape((roi.shape[0] * roi.shape[1], 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
colourQuantisedImage = res.reshape((roi.shape))

# meanShiftedImage = cv2.pyrMeanShiftFiltering(res2, 5, 60, 3)
# median = cv2.medianBlur(meanShiftedImage, 5)
bgr = cv2.cvtColor(colourQuantisedImage,cv2.COLOR_LAB2BGR)

gray = cv2.cvtColor(colourQuantisedImage,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cannyEdges = cv2.Canny(thresh, 100, 200)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closedEdges = cv2.morphologyEx(cannyEdges, cv2.MORPH_CLOSE, kernel)

sure_bg = cv2.dilate(closedEdges, kernel,iterations=3)

plt.subplot(121),plt.imshow(sure_bg, cmap = 'gray')
plt.title('K Means Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(clonedRoi, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

# clt = KMeans(5)
#
# clt.fit(image)
#
# hist = utils.centroid_histogram(clt)
# colours = utils.main_colours(hist, clt.cluster_centers_)
#
# bar = utils.plot_colors(hist, clt.cluster_centers_)
#
# bounds = utils.BGR2HSVColourBoundaries(colours[2], 15)
#
# mask = cv2.inRange(hsv, bounds[0], bounds[1])
# res = cv2.bitwise_and(hsv, hsv, mask = mask)

cannyEdges = cv2.Canny(colourQuantisedImage, 100, 200)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closedEdges = cv2.morphologyEx(cannyEdges, cv2.MORPH_CLOSE, kernel)

_, contours, _ = cv2.findContours(cannyEdges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	cv2.drawContours(roi, [approx], -1, (0, 255, 0), 2)

cv2.imshow("contours", closedEdges)
cv2.waitKey()

# plt.subplot(121),plt.imshow(meanShiftedImage2,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(cannyEdges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
