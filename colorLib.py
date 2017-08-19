import cv2 as cv
import numpy as np

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

def showFace(shading, skin, opacity, csm) :
    """
    shading is a numpy array capturing the variation in light intensity on the face.
    It is 1.0 where the facial color is equal to the skin reflectance.
    skin is a BGR numpy array showing linear skin reflectance normalized 0.0 to 1.0
    opacity is the opacity of the cosmetic layer (0 is transparent, 3 or 4 is pretty opaque)
    cosmetic is the [B G R] tuple with the reflectance of a very thick layer of cosmetic
    It returns a linear BGR numpy image of the face with cosmetic applied normalized 0.0 to 1.0
    """
    opF = np.exp(-opacity)   # F in notes
    var = np.copy(skin)
    G = np.copy(skin)
    for idx in [0,1,2] :
        var[:,:,idx] = var[:,:,idx] - csm[idx]  # delta in notes
        G[:,:,idx] = var[:,:,idx] * csm[idx] + csm[idx] * csm[idx] - 1  # G in notes
        G[:,:,idx] =(var[:,:,idx] * opF - G[:,:,idx] * csm[idx])/(var[:,:,idx] * opF * csm[idx] - G[:,:,idx])  # R in notes
        G[:,:,idx] = G[:,:,idx] * shading  # skin color = reflectance * shading
    return  G







