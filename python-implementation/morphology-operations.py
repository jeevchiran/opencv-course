import numpy as np
import cv2

img = cv2.imread('../Resources/Photos/cats.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# Simple Thresholding
threshold, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('Simple Thresholded', thresh)

# threshold, thresh_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV )
# cv2.imshow('Simple Thresholded Inverse', thresh_inv)


# dilation
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5))
# dilated_image = cv2.filter2D(thresh,-1,dilation_kernel)
dilated_image = cv2.morphologyEx(
    thresh, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
# cv2.imshow('eroded image', dilated_image)


# erosion
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (5, 5))
# erosion_image = cv2.filter2D(thresh,-1,erosion_kernel)
erosion_image = cv2.morphologyEx(
    thresh, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
# cv2.imshow('eroded image', erosion_image)

# open (erosion + dilation)
# open = cv2.filter2D(erosion_image,-1,dilation_kernel)
open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
cv2.imshow('open', open)

# close (dilation + erosion)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
cv2.imshow('close', closed)

# gradient (dilate-erosion) edge detection
gradient = cv2.subtract(dilated_image, erosion_image)
# gradient = cv2.morphologyEx(thresh,cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
cv2.imshow('gradient', gradient)


# top hat (image - open) remove main subject of image highlight minor details
top_hat = cv2.subtract(thresh, open)
top_hat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
cv2.imshow('top hat', top_hat)

# black hat (image - close) highlight main subject i.e. bright object in dark background
black_hat = cv2.subtract(thresh, closed)
black_hat = cv2.morphologyEx(
    thresh, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
cv2.imshow('black hat', black_hat)

cv2.waitKey(0)
