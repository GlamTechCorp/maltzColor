import cv2 as cv
import numpy as np
import colorLib as clib
# import sys as sys   # so can uses sys.exit() for diagnostic purposes

""" This shows what an eyelid would look like with eye shadow. """

    # set parameters
Yskin = 0.5  # a guess at average skin neutral reflection
imageName = 'eye1.bmp'  # the original eye region image
ffName = 'floodfill.bmp'  # the flood filled eye lid region
ffVal = 200   # the value of the pixels in the flood filled region
featheredName = 'feathered.bmp'  # the eye lid region with eye shadow feathering and enclosing line
eyeShadow = [0.5, 0.3, 0.3]   # non textured eye shadow BGR reflectance (not used if textureName is set)
textureName = 'texture.bmp'  # textured eye shadow image
# textureName = ''  # textureName not set
opacity = .5   # opacity of the thickest part of the eye shadow application

# Simplified extraction of luminance variation and skin reflectance from an image
    # get face RGB reflectance and luminance
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

    # make Yshad (amount of shadow) and refAve (average skin reflectance) consistent with the image
shading = np.array(gray/(255.0 * Yskin))  # conforms to Yshad definition in notes
faceR = np.array(face/255.0)  # conform to RGBface definition in notes
eyelid = cv.imread(ffName, 0)
mask = (eyelid == ffVal)   # picks out flood filled region
prodGbr = [[],[],[]]
refAve = [1,1,1]
maxRgb = [1,1,1]
prodY = mask * shading
sumY = prodY.sum()
for idx in [0,1,2] :   # calculated Y weighted average skin color
    prodGbr[idx] = mask * faceR[:,:,idx]
    refAve[idx] = prodGbr[idx].sum() / sumY
    maxRgb[idx] = shading.max() * refAve[idx]
print 'average skin reflectance (bgr)', refAve
print 'shading max', shading.max()

    # prevent roll overs (shading * refAve > 255).
print 'maxRgb before limiting', maxRgb
for idx in [0,1,2] :
    if maxRgb[idx] < 1.0 :
        maxRgb[idx] = 1.0
print 'maxRgb after limiting', maxRgb
for idx in [0,1,2]:
    refAve[idx] = refAve[idx] / maxRgb[idx]
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

""" prepare thickness input array for showFaceGen() """
thickness = cv.imread(featheredName, 0)
mask = (thickness < 255)  # eliminate border line
thickness = thickness * mask
cv.imshow('thickness', thickness)
thickness = thickness/255.0   # convert to 0 to 1 encoding

""" prepare cosmetic array for showFaceGen()"""
if len(textureName) > 4 :   # textured version will be used
    csm = cv.imread(textureName)
    print 'texture file', textureName, 'read'
    csm = csm / 255.0    # change to 0-1 encoding
else :     # non textured version
    csm = np.copy(faceR)   # color of cosmetic
    for idx in [0,1,2] :
        csm[:,:,idx] = eyeShadow[idx]  # not equal to refAve on purpose
""" keep 0 < csm < 1 """
csm = csm + 0.01
maxcsm = csm.max()  # reflectance >= 1.0 give 0/0 condition in calculation
if maxcsm > 0.99 :
    maxcsm = maxcsm+0.02
    csm = csm / maxcsm
print 'min csm ', csm.min()
print 'max csm', csm.max()

# sys.exit()  # diagnostic

# use cosmetic with eye shadow of varying thickness
# note this is not exactly right since scattering per unit thickness varies in this case
# because the scattering centers are not scattered uniformly throughout the bulk of the
# material, but concentrated into large particles.
cosmeticRGB = clib.showFaceGen(shading, faceR, thickness*opacity, csm)

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
