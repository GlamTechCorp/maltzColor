import cv2 as cv
import numpy as np
import collections as col

""" Note linear RGB is defined in terms of the sRGB phosphors """

def getPatchColor(patch) :

    """
    patch should be an sRGB image with the correct white point.  In general, it will be
    captured from a web page with a sRGB = 255,255,255 background, so no further
    white point adjustment will be necessary.  It is assumed that at least some of the
    pixels in the image show the true color of a very thick layer of the cosmetic.

    It will return a [B G R] tuple (OpenCV ordering) for the reflectance of the cosmetic
    applied in a very thick layer, which is the parameter that is needed for a Kubelka-Munk
    calculation of the appearance of skin with the cosmetic applied in any thickness.
    The normalization will be 0.0 to 1.0.
    """

    dark = 1000.0
    darkest = [-1.0, -1.0, -1.0]
    for row in patch:
        for col in row :
            bR = toReflectance(col[0])
            gR = toReflectance(col[1])
            rR = toReflectance(col[2])
            dd = 0.33 * rR + 0.66 * gR + 0.07 * bR
            if dd < dark :
                dark = dd
                darkest = [bR, gR, rR]
    return darkest

def fromReflectance(ref) :
    """
    ref is a 0 to 1.0 reflectance value
    returns a 0 to 255 sRGB value
    assumes a reflectance of 1.0  corresponds to an sRGB of 255
    """
    fac = 0.055
    if ref < 0.0031308 :
        return 255 * 12.92 * ref
    else :
        return 255 * ((1+fac) * ref ** (1/2.4) - fac)

def toReflectance(col) :
    """
    col is an sRGB value 0 to 255 normalization
    returns a reflection 0.0 to 1.0 normalization
    assumes sRGB 255 corresponds to a reflection of 1.0
    """
    ref = col/255.0
    fac = 0.055
    if ref < 0.04045 :
        ref = ref/12.92
    else :
        ref = ((ref + fac) / (1+fac)) ** 2.4
    return ref

def showFace(shading, skin, thickness, csm) :
    """
    shading is a numpy array capturing the variation in light intensity on the face.
    It is 1.0 where the facial color is equal to the skin reflectance.
    skin is a BGR numpy array showing linear skin reflectance normalized 0.0 to 1.0
    thickness is the thickness of the cosmetic layer (0 is transparent, 3 or 4 is pretty opaque)
    csm is the [B G R] tuple with the reflectance of a very thick layer of cosmetic
    It returns a linear BGR numpy image of the face with cosmetic applied normalized 0.0 to 1.0
    """
    opF = [1.0, 1.0, 1.0]
    for idx in [0, 1, 2] :
        opF[idx] = thickness * (1 - csm[idx]*csm[idx]) / csm[idx]
        opF[idx] = np.exp(-opF[idx])   # F in notes
    var = np.copy(skin)
    G = np.copy(skin)
    for idx in [0,1,2] :
        var[:,:,idx] = var[:,:,idx] - csm[idx]  # delta in notes
        G[:,:,idx] = var[:,:,idx] * csm[idx] + csm[idx] * csm[idx] - 1  # G in notes
        G[:,:,idx] =(var[:,:,idx] * opF[idx] - G[:,:,idx] * csm[idx])/(var[:,:,idx] * opF[idx] * csm[idx] - G[:,:,idx])  # R in notes
        G[:,:,idx] = G[:,:,idx] * shading  # skin color = reflectance * shading
    return  G

def showFaceGen(shading, skin, thickness, csm) :
    """
    shading is a numpy array capturing the variation in light intensity on the face.
    It is 1.0 where the facial color is equal to the skin reflectance.
    skin is a BGR numpy array showing linear skin reflectance normalized 0.0 to 1.0
    thickness is a numpy array with the image-wise thickness of the cosmetic layer
    (0 is transparent, 3 or 4 is pretty opaque)
    csm is a BGR numpy array with the reflectance of a very thick layer of cosmetic
    This captures possible cosmetic texture
    It returns a linear BGR numpy image of the face with cosmetic applied normalized 0.0 to 1.0
    """
    opF = np.copy(skin)
    npthick = np.copy(thickness)
    for idx in [0, 1, 2]:
        opF[:, :, idx] = npthick * (1 - csm[:, :, idx] * csm[:, :, idx]) / csm[:, :, idx]
        opF[:, :, idx] = np.exp(-opF[:, :, idx])  # F in notes
    var = np.copy(skin)
    G = np.copy(skin)
    tmp = np.copy(skin)
    for idx in [0,1,2] :
        var[:,:,idx] = var[:,:,idx] - csm[:,:,idx]  # delta in notes
        G[:,:,idx] = var[:,:,idx] * csm[:,:,idx] + csm[:,:,idx] * csm[:,:,idx] - 1  # G in notes
        tmp[:,:,idx] = var[:,:,idx] * opF[:,:,idx] * csm[:,:,idx] - G[:,:,idx]
        G[:,:,idx] =(var[:,:,idx] * opF[:,:,idx] - G[:,:,idx] * csm[:,:,idx])/(var[:,:,idx] * opF[:,:,idx] * csm[:,:,idx] - G[:,:,idx])  # R in notes
        G[:,:,idx] = G[:,:,idx] * shading  # skin color = reflectance * shading
    return  G

def floodfill(img, pt, val) :
    """ img is a 0-255 monochrome image, 0 everywhere except at the boundary of the area to be filled with
        the constant alue val.  pt (row, column) is the starting point for the flood fill process.
        img is modified. """
    qq = col.deque()
    img[pt[0], pt[1]] = val
    addAdj(img, qq, val, pt)
    while len(qq)> 0 :
        center = qq.popleft()
        addAdj(img, qq, val, center)

""" floodfillFunc calls the getVal method of the two classes below
    to fill an area with constant or non-constant values """

class constVal() :
    """ Done this more complicated way to be consistent with featheredVal class """
    def setVal(self, val):  # call this before calling getVal
        self.val = val
    def getVal(self, pt):  # pt is (row, col)
        return self.val   # this class returns a constant value

class featheredVal(constVal) :   # an extension of constVal that produces a feathered eye shadow thickness function
    def getLine(self, imfile):
        """ finds the pixels in the line above the eyelid and puts them in self.line
            imfile contains only the line at the top of the eyelid (value = 255) and 0 everywhere else
            call this function before calling floodfillFunc
        """
        srch = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
        img = cv.imread(imfile, 0)
        self.line = []
        self.val = []
        nrow = 0
        for rows in img :
            ncol = 0
            for cols in rows :
                if cols == 255 :
                    self.line.append((nrow, ncol))
                    for adj in srch :
                        img[adj[0]+nrow, adj[1]+ncol] = 0   # cuts number of points in line
                                                        # which speeds up minval calculation in getVal
                    # does not check for out of image pixels which normally won't be a problem
                ncol = ncol + 1
            nrow = nrow + 1
        print(len(self.line))  # a diagnostic
    def getVal(self, pt):
        """ calculate an image value for each pixel during floodfillFunc execution
            The value is high at the bottom of the eyelid and goes to zero at the top
        """
        pwr = 0.3  # adjusts sharpness of drop to zero at top of eyelid - this value seems to work pretty well
        minval = (self.line[0][0]-pt[0])**2 + (self.line[0][1]-pt[1])**2
        for ptl in self.line :  # gets the minimum distance between pt and line
            dist = (ptl[0] - pt[0])**2 + (ptl[1] - pt[1])**2
            if dist < minval :
                minval = dist
        minval = minval ** pwr
        self.val.append((pt, minval))  # ((y,x),minval)  # val contains the non-normalized eye shadow thickness values
        return 100  # this sets the pixels in the flood filled image area - any number > 0 will do
    def renorm(self, img):
        """ re-normalize values in eyelid thickness image to go from 0 to 254
            put them in img which could start off = 0 everywhere
            call this after calling floodfillFunc """
        maxval = 0
        for val in self.val :
            if maxval < val[1] :
                maxval = val[1]
        for val in self.val :
            pt = val[0]
            val1 = val[1]
            img[pt[0], pt[1]] = 254 * val1/maxval

def floodfillFunc(img, pt, func) :
    """ img is a 0-255 monochrome image, 0 everywhere except at the boundary of the area to be filled.
        This boundary is set to a value of 255.  It will be filled with values between 0 and 254 generated
        by the getVal(pt) method of func, which is either constVal or featheredVal.
        pt (row, column) is the starting point for the flood fill process.
        img is modified. """
    qq = col.deque()  # this is a more sophisticated list
    val = func.getVal(pt)
    img[pt[0], pt[1]] = val
    addAdj(img, qq, val, pt)
    while len(qq)> 0 :
        center = qq.popleft()
        addAdj(img, qq, func.getVal(center), center)

def addAdj(img, qq, val, center) :
    """ add img[:,:] == 0 points adjacent to center to qq,
        and set corresponding image points to val.
        Does not check for points out of image
        which normally will not be a problem.
    """
    srch = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]
    shifted = (0,0)
    for shift in srch :
        shifted = (center[0]+shift[0], center[1]+shift[1])
        if img[shifted[0], shifted[1]] > 0 :
            continue
        img[shifted[0], shifted[1]] = val
        qq.append(shifted)    # strange things happen if you append [] to ()

""" get CIELAB difference """

def fLAB(t) :
    """ this defines the CIELAB nonlinearity"""
    delta = 6.0/29
    delta2 = delta*delta
    delta3 = delta*delta2
    if t > delta3 :
        return t ** 0.33333333
    else :
        val = 4.0/29 + t / (3 * delta2)
        return val

def RGBtoLAB(BGR) :
    """ Converts linear BGR 0-1 normalization, sRGB phosphors,
    to CIELAB with a D65 illuminant and 0-100 normalization"""
    # first convert BGR to RGB to XYZ
    RGB = [BGR[2], BGR[1], BGR[0]]
    M = [[0.4124, 0.3576, 0.1805],
         [0.2126, 0.7152, 0.0722],
         [0.0193, 0.1192, 0.9505]] # RGB to XYZ
    XYZ = np.matmul(M, RGB)
    # XYZ to L*a*b* with D65 normalization
    Xn = 0.95047
    Yn = 1.000
    Zn = 1.08883
    LAB = [1.0,1.0,1.0]
    LAB[0] = 116 * fLAB(XYZ[1] / Yn) - 16
    xyDif = fLAB(XYZ[0] / Xn) - fLAB(XYZ[1] / Yn)
    yzDif = fLAB(XYZ[1] / Yn) - fLAB(XYZ[2] / Zn)
    LAB[1] = 500 * xyDif
    LAB[2] = 200 * yzDif
    return LAB

def difLAB(BGR1, BGR2) :
    """ calculates CIELAB distance, 0-100 normalization between two colors
    BGR normalization is 0-1 """
    LAB1 = RGBtoLAB(BGR1)
    LAB2 = RGBtoLAB(BGR2)
    dLAB = (LAB1[0] - LAB2[0]) **2 +(LAB1[1] - LAB2[1]) **2 +(LAB1[2] - LAB2[2]) **2
    dLAB = dLAB ** 0.5
    return dLAB













