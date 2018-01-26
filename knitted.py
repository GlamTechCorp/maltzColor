import cv2 as cv
import numpy as np

pname = '20171106_120437a.jpg'  # unscaled image of textured eye shadow
iname = 'eye1.bmp'    # selfie of face
scale = 0.62  # number of pixels/inch in selfie divided by number of pixels/inch in eye shadow image

patch = cv.imread(pname)
img = cv.imread(iname)

# rescale patch to resolution of image
shp = np.shape(patch)
prow = shp[0]
pcol = shp[1]
patch = cv.resize(patch, (int(scale*prow), int(scale*pcol)))
shp = np.shape(patch)
prow = shp[0]
pcol = shp[1]

# make flipped versions
r0 = cv.flip(patch,0)
rp = cv.flip(patch,1)
rm = cv.flip(patch,-1)

# get image information
shpi = np.shape(img)
irow = shpi[0]
icol = shpi[1]
nrow = int(irow/prow)
ncol = int(icol/pcol)

print 'patch shape (rows and columns)', prow, pcol
print 'image shape (rows and columns)', irow, icol
vflip = 1
for idy in range(nrow+1):
    hflip = 1
    for idx in range(ncol+1):
        slice = img[idy*prow:(idy+1)*prow, idx*pcol:(idx+1)*pcol]
        slshape = np.shape(slice)
        slrows = slshape[0]
        slcols = slshape[1]
        if vflip == 1 and hflip == 1:
            slice[:,:,:] = patch[0:slrows,0:slcols,:]
        if vflip == -1 and hflip == 1 :
            slice[:,:,:] = r0[0:slrows,0:slcols,:]
        if vflip == 1 and hflip == -1 :
            slice[:,:,:] = rp[0:slrows,0:slcols,:]
        if vflip == -1 and hflip == -1 :
            slice[:,:,:] = rm[0:slrows,0:slcols,:]
        hflip = -1 * hflip
    vflip = -1*vflip

cv.imshow('img', img)
cv.imwrite('texture.bmp', img)
cv.waitKey(0)