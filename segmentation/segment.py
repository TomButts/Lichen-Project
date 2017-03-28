import numpy as np
import cv2
from matplotlib import pyplot as plt
from crop import Crop
import grab_cut
import k_means

img = cv2.imread("D:\Python\Lichen\images\hypogymnia.jpg")

# img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)

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
roi = img[cropWindow.refPt[0][1]:cropWindow.refPt[1][1], cropWindow.refPt[0][0]:cropWindow.refPt[1][0]]

# clone roi to compare
clonedRoi = roi.copy()

# filters
roi = cv2.pyrMeanShiftFiltering(roi, 15, 60, 3)
roi = cv2.medianBlur(roi, 5)
roi = cv2.pyrMeanShiftFiltering(roi, 5, 60, 3)

# Can be useful to help pick a K value
# cv2.imshow("mean shifted", roi)
# cv2.waitKey()

colourQuantisedImage = k_means.cluster_colours(roi, 2)

# segmentedImage = grab_cut.grab_cut(clonedRoi)

# OpenCV Operates in BGR so convert to RGB for matplotlib plots
segmentedImage = cv2.cvtColor(segmentedImage, cv2.COLOR_BGR2RGB)
clonedRoi = cv2.cvtColor(clonedRoi, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(segmentedImage)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(clonedRoi)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
