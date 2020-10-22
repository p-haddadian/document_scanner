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
gray_image = cv2.GaussianBlur(gray_image, (13, 13), 0)
edged = cv2.Canny(gray_image, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Original image", image)
cv2.imshow("Edge detected image", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# finding the contours in the edged image, keeping only the largest ones, and initialize the screen contour
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key= cv2.contourArea, reverse= True)

# loop over the contours
for c in contours:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has 4 points,
    # then we can assume that we found the paper
    print(len(approx))
    if(len(approx) == 4):
        screenCnt = approx
        break

# show the contour of the piece of paper
print("STEP 2: Find Contours of the paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Contoured Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it to give it that black and white paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 17, 'gaussian', 10)
warped = (warped > T).astype('uint8') * 255

# show the original and the scanned image
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)