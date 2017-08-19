import cv2 as cv
import numpy as np
import colorLib as clib

# Find color of the thickest cosmetic (the darkest color)
patch = cv.imread('armaniPatch.png', 1)
darkest = clib.getPatchColor(patch)
print 'darkest cosmetic patch reflectance (bgr)', darkest

# Simplified extraction of luminance variation and skin reflection from an image
Yskin = 0.5
imageName = 'crop4.jpg'
face = cv.imread(imageName, 1)
gray = cv.imread(imageName, 0)
idy = 0   # first transform from sRGB to RGB and get luminance
for row in face:
    idx = 0
    for col in row:
        bR = clib.toReflectance(col[0])  # sRGB(0 to 255) to RGB (0.0 to 1.0)
        gR = clib.toReflectance(col[1])
        rR = clib.toReflectance(col[2])
        face[idy,idx] = [bR*255, gR*255, rR*255] # now RGB instead of sRGB, * 255 to make it viewable
        gray[idy,idx] = 255 * (0.33 * rR + 0.66 * gR + 0.07 * bR) # * 255 to make it viewable
        idx = idx+1
    idy = idy+1
# cv.imshow('gray intensity', gray)  # Y viewed with sRGB TRC
# cv.imshow('RGB intensity', face)  # RGB viewed with sRGB TRC
# cv.waitKey(0)  # showing linear Y and RGB with an sRGB TRC often makes it too dark to see

# make Yshad and refAve consistent with the image
shading = np.array(gray/(255.0 * Yskin))  # conforms to Yshad definition in notes
print 'shading range', shading.min(), 'to', shading.max()
faceR = np.array(face/255.0)  # conform to RGBface definition in notes
facesum = faceR.sum(axis=0)  # sum each row
facesum = facesum.sum(axis=0)  # sum the sums of each row
sumshad = shading.sum()  # sum of Yshad
refAve = facesum/sumshad
print 'average skin reflectance (bgr)', refAve

# prevent roll overs.
maxRgb = shading.max() * refAve
print 'maxRgb before limiting', maxRgb
for idx in [0,1,2] :
    if maxRgb[idx] < 1.0 :
        maxRgb[idx] = 1.0
print 'maxRgb after limiting', maxRgb
refAve = refAve / maxRgb
print 'refAve after limiting', refAve

# make and show simulated bare face
idy = 0
for row in shading:
    idx = 0
    for col in row:
        faceR[idy,idx] = refAve
        bR = clib.fromReflectance(col * refAve[0])
        gR = clib.fromReflectance(col * refAve[1])
        rR = clib.fromReflectance(col * refAve[2])
        face[idy,idx] = [bR, gR, rR] # now simulated sRGB
        idx = idx+1
    idy = idy+1
cv.imshow('simulated bare face', face)
cv.imwrite('simulation.png', face)

# add pimples
faceR[50:70, 200:220, : ] = faceR[50:70, 200:220, : ] + [0, 0, 0.3]
idy = 0   # go from RGB to sRGB
for row in faceR:
    idx = 0
    for col in row:
        bR = clib.fromReflectance(col[0] * shading[idy,idx])
        gR = clib.fromReflectance(col[1] * shading[idy,idx])
        rR = clib.fromReflectance(col[2] * shading[idy,idx])
        face[idy,idx] = [bR, gR, rR] # now simulated sRGB
        idx = idx+1
    idy = idy+1
cv.imshow('with pimple', face)
cv.imwrite('withPimple.png', face)

# use cosmetic with thick layer reflectance matching average skin reflection
opacity = 2.0
cosmetic = refAve
cosmeticRGB = clib.showFace(shading, faceR, opacity, cosmetic)

# prevent roll overs.
for idx in [0,1,2] :
    maxRgb[idx] = cosmeticRGB[:, idx].max()
print 'cosmetic image maxRgb before limiting', maxRgb
for idx in [0,1,2] :
    if maxRgb[idx] < 1.0 :
        maxRgb[idx] = 1.0
print 'cosmetic image maxRgb after limiting', maxRgb

# show face with cosmetics
idy = 0
for row in cosmeticRGB:
    idx = 0
    for col in row:
        bR = clib.fromReflectance(col[0]/maxRgb[0])
        gR = clib.fromReflectance(col[1]/maxRgb[1])
        rR = clib.fromReflectance(col[2]/maxRgb[2])
        face[idy,idx] = [bR, gR, rR] # now simulated sRGB
        idx = idx+1
    idy = idy+1
cv.imshow('face with cosmetic', face)
cv.imwrite('withCosmetic.png', face)
cv.waitKey(0)
