import os
import numpy as np
import skimage.io as io
import pickle

datasetPath = "C:/Users/DiogoNunes/Documents/Tesselo/data/tesselo-training-tiles"
pixels = dict()
for i in range(47):
    pixels[i] = 0
pixels[99] = 0

tileXList = os.scandir(datasetPath)
for tileX in tileXList:
    tileYList = os.scandir(tileX)
    for tileY in tileYList:
        imgCOS_GT = io.imread(os.path.join(datasetPath, tileX.name, tileY.name, "COS.tif"), as_gray = True)
        for i in range(47):
            pixels[i] += np.count_nonzero(imgCOS_GT == i)
        pixels[99] += np.count_nonzero(imgCOS_GT == 99)
with open('pixels.txt', 'ab') as pix_file:
    pickle.dump(pixels, pix_file)
