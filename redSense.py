import cv2 as cv
import numpy as np
import colorLib as clib

""" This is for making a mask from an image with an area encircled with a red line. """

fname = 'eye1feather.bmp'
maskname = 'eye1featherLine.bmp'
img = cv.imread(fname)
gray = cv.imread(fname, 0)
red = cv.imread(fname)

red = img[:,:] == [0,0,255] # wherever img[:,:,0] is 0 red[:,:,0] is 1, and similar for other seps

print ('img', np.shape(img))
print ('red', np.shape(red))
print ('gray', np.shape(gray))

gray[:,:] = 255 * (red[:,:,0] & red[:,:,1] & red[:,:,2])
cv.imshow('eyelid', gray)
cv.imwrite(maskname, gray)

cv.waitKey(0)

