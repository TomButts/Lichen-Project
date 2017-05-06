import numpy as np
import cv2
from matplotlib import pyplot as plt
from crop import Crop
from colour_picker import ColourPicker
import grab_cut
import k_means

img = cv2.imread("D:\Python\Lichen\images\lichen2.jpg")

img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)

cv2.namedWindow('Lichen Image')

clonedImage = img.copy()

# Use a copy so roi can be drawn then thrown away
cropWindow = Crop(clonedImage)

roi = cv2.setMouseCallback('Lichen Image', cropWindow.crop)

while True:
	# display the image and wait for a keypress
	cv2.imshow("Lichen Image", cropWindow.imageToCrop)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		img = clonedImage.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

cv2.destroyAllWindows()

# coords are y1,y2, x1,x2 format
original_roi = img[cropWindow.refPt[0][1]:cropWindow.refPt[1][1], cropWindow.refPt[0][0]:cropWindow.refPt[1][0]]
roi = img[cropWindow.refPt[0][1]:cropWindow.refPt[1][1], cropWindow.refPt[0][0]:cropWindow.refPt[1][0]]

# clone roi to compare
clonedRoi = roi.copy()

# filters
roi = cv2.pyrMeanShiftFiltering(roi, 15, 60, 3)
# roi = cv2.medianBlur(roi, 5)
roi = cv2.pyrMeanShiftFiltering(roi, 5, 60, 3)

roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
clonedRoi = cv2.cvtColor(clonedRoi, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(clonedRoi)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(roi)
plt.title('Posterised Image'), plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)

# Can be useful to help pick a K value
# cv2.imshow("mean shifted", roi)
# cv2.waitKey()

colourQuantisedImage = k_means.cluster_colours(roi, 2)

colourWindow = ColourPicker(colourQuantisedImage)

cv2.namedWindow('Select Region')

cv2.setMouseCallback('Select Region', colourWindow.pick_colour)

while True:
	# display the image and wait for a keypress
	cv2.imshow("Select Region", colourWindow.image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		img = clonedImage.copy()
	elif key == ord("p"):
		print(colourWindow.pixel_values)
		print(colourWindow.pixel_coords)
	elif key == ord("c"):
		break

lower = np.array(colourWindow.pixel_values, dtype = "uint8")
upper = np.array(colourWindow.pixel_values, dtype = "uint8")

mask = cv2.inRange(colourQuantisedImage, lower, upper)
output = cv2.bitwise_and(colourQuantisedImage, colourQuantisedImage, mask = mask)

# cv2.imshow('mask', output)
# cv2.waitKey(0)

gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

(thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

rgb_quantised = cv2.cvtColor(colourQuantisedImage, cv2.COLOR_BGR2RGB)
rgb_bw = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2RGB)

plt.subplot(121),plt.imshow(rgb_quantised)
plt.title('Colour Quantised Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(rgb_bw)
plt.title('Binarised Image'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)

segmented_im = cv2.bitwise_and(original_roi, original_roi, mask = im_bw)

cv2.imshow('seg', segmented_im)
cv2.waitKey(0)

# segmentedImage = grab_cut.grab_cut(clonedRoi)

# OpenCV Operates in BGR so convert to RGB for matplotlib plots
segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_BGR2RGB)
original_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(original_roi)
plt.title('Region of Interest'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(segmented_im)
plt.title('Segmented RoI'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
