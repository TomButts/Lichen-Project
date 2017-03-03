import cv2

class Crop:
	refPt = []
	cropping = False

	def __init__(self, image):
		self.imageToCrop = image

	def crop(self, event, x, y, flags, param):
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			self.refPt = [(x, y)]
			self.cropping = True

		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# record the ending (x, y) coordinates and indicate that
			# the cropping operation is finished
			self.refPt.append((x, y))
			self.cropping = False

			# draw a rectangle around the region of interest
			cv2.rectangle(self.imageToCrop, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
			cv2.imshow("lichen", self.imageToCrop)
