from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt
import UtilityFunctions as utils
from Segmentation.Crop import Crop

img = cv2.imread("D:\Python\Lichen\images\lichen2.jpg")

img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)

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

#filters
roi = cv2.pyrMeanShiftFiltering(roi, 15, 60, 3)
roi = cv2.medianBlur(roi, 5)
roi = cv2.pyrMeanShiftFiltering(roi, 5, 60, 3)

cv2.imshow("mean shifted", roi)
cv2.waitKey()

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

mask = np.zeros(colourQuantisedImage.shape[:2], np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

height, width, channels = colourQuantisedImage.shape

# x,y,width,height
rect = (1,1,width,height)

cv2.grabCut(colourQuantisedImage, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
segmentedImage = clonedRoi * mask2[:,:,np.newaxis]

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
