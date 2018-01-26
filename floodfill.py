import cv2 as cv
import numpy as np
import colorLib as clib

""" This is a driver for the flood fill library modules. """

fname = 'eyelid.bmp'  # this image has upper and lower boundary lines that surround the entire eyelid
fupper = 'eye1featherLine.bmp' # this image only contains the upper boundary line
# both are single separation images.  The pixels in the line are 255; all the others are zero
pt = (144,269)   # seed within the eyelid
img = cv.imread(fname, 0)
img2 = np.copy(img)
img3 = np.copy(img)

clib.floodfill(img, pt, 200)
cv.imshow('floodfill', img)
cv.imwrite('floodfill.bmp', img)

# illustrates how to do a constant flood fill with floodFillFunc
val = clib.constVal()   # make a constVal object
val.setVal(100)
clib.floodfillFunc(img2, pt, val)
cv.imshow('floodfillFunc', img2)  # this image is not used

# illustrates how to do a variable flood fill with floodFillFunc
val = clib.featheredVal()  # make a featheredVal (an extension of constVal) object.
val.getLine(fupper)
clib.floodfillFunc(img3, pt, val)
val.renorm(img3)
cv.imshow('floodfillFeathered', img3)
cv.imwrite('feathered.bmp', img3)

cv.waitKey(0)