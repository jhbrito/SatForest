import keras
from keras.models import load_model
import numpy as np
import os
from data import *

datasetPath = 'C:\\Users\DiogoNunes\Documents\Tesselo\data\\tesselo-training-tiles'
modelsPath = 'C:\\Users\DiogoNunes\Documents\Tesselo\\unet\models'
dataStatsFile = 'C:\\Users\DiogoNunes\Documents\Tesselo\data\data_stats.txt'
modelFilePath = os.path.join(modelsPath, 'unet_COS.hdf5')
tile = '7758\\6237'
input_size = (256, 256)
with open(dataStatsFile, "rb") as fp:
    print("Loading data statistics")
    dataStats = pickle.load(fp)

# load model
model = load_model(modelFilePath)
# summarize model
model.summary()

imgs = getImgs(datasetPath, tile, dataStats, use_max = False)

img = np.zeros(input_size + (len(channels),))
for i in range(len(imgs)):
    img[:, :, i] = imgs[i]

img = np.array([img])
