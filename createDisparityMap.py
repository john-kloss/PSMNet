import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

topDir = os.getcwd() + '/data/top/'
bottomDir = os.getcwd() + '/data/bottom/'
disparityDir = os.getcwd() + '/data/disparity/'

for im in os.listdir(topDir):
    imgL = cv2.imread(topDir+im, 0)
    imgR = cv2.imread(bottomDir+im, 0)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    print(im)
    newPath = disparityDir+im
    cv2.imwrite(newPath.replace('.jpg', '.pfm'), disparity)
    # plt.imshow(disparity, 'gray')
    # plt.show()
