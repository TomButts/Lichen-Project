import cv2

class ColourPicker:
	pixel_values = []
	pixel_coords = []

	def __init__(self, image):
		self.image = image

	def pick_colour(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.pixel_coords = [x , y]
			self.pixel_values = self.image[x,y]
