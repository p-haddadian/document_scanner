# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required= True, help= 'path to the image to be scanned')
args = vars(ap.parse_args())

# load the image and calculate the ratio between the old height and the new height, clone it, and resize it
image = cv2.imread(args['image'])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# convert the image into grayscale, blur it, and find edges.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edged = cv2.Canny(gray_image, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Original image", image)
cv2.imshow("Edge detected image", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()