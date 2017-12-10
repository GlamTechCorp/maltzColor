import cv2 as cv
import colorLib as clib
import os as os
import tkFileDialog as tkd

print('\nUse the file browser go to the directory with the cosmetic patch images of interest.')
print('Choose any patch and the program will make a file called rgb.txt in this directory.')
print('The first line of this file will contain the directory path.')
print('The rest of this file will contain cosmetic names and RGB reflectances for all the cosmetic patches')

fname = tkd.askopenfilename()  # directoy path and filename
idx = fname.rfind('/')
imdir = fname[0:idx+1]  # directory path
fo = open(imdir+'rgb.txt', 'w')  # open output file
val = imdir + '\n'
fo.write(val)
lst = os.listdir(imdir)
for line in lst :
    idx = line.rfind('.jpg')
    if idx < 0 :
        continue  # skip non-image files
    img = cv.imread(imdir+line)
    bgr = clib.getPatchColor(img)
    line = line[0:idx]
    val = line + '  R ' + str(bgr[2]) + '  G ' + str(bgr[1]) + '  B ' + str(bgr[0]) + '\n'
    fo.write (val)


