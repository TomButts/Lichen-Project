import numpy as np
import cv2

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX

	# return the bar chart
	return bar

def main_colours(hist, centroids):
	colours = []
	# Add main colours to array
	for (percent, colour) in zip(hist, centroids):
		colours.append(colour)

	colours = np.array(colours)

	return colours

def BGR2HSVColourBoundaries(colour, range):

	lowerBound = list(colour)
	upperBound = np.array([180, 255, 255])

	lowerBound[:] = [value - range for value in lowerBound]

	# upperBound[:] = [value - range for value in upperBound]

	# convert to format that allows cv conversion to be applied
	lowerBound = cv2.cvtColor(np.uint8([[lowerBound]]), cv2.COLOR_BGR2HSV)
	upperBound = cv2.cvtColor(np.uint8([[upperBound]]), cv2.COLOR_BGR2HSV)

	print(lowerBound)
	print(upperBound)

	return [lowerBound, upperBound]
