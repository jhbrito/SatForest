import sys
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import qVersion, QUrl
from PyQt5 import QtWebEngine
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QVBoxLayout

import skimage.io as io
import numpy as np
import tensorflow as tf
from COS_train_options import channels, class_labels, class_aggregation_COLOR_DICT
from data import classAgregateCOS, labelVisualizeCOS
from data import getImgs as data_getImgs
from model_unet_COS import unetL5
import pickle

print("Tensorflow {}".format(tf.__version__))
if tf.test.is_gpu_available():
    print("GPU available: {}".format(tf.test.gpu_device_name()))
else:
    print("GPU not available")


dataset_path = "C:/Tesselo/data/tesselo-training-tiles"
models_path = "./models"
unet_level = 5
modelFileName="unetCOSV1_ClassesCOSN1B_NClasses4_Level5_Featuremaps64_Padsame_BNNone_TCFalse_Dropout0.5_Batchsize4_Epochs100_Datetime20210522-015747.hdf5"
data_stats_file = "./data/data_stats_V1.txt"

with open(data_stats_file, "rb") as fp:
    print("Loading training data statistics")
    dataStats = pickle.load(fp)

num_class = len(class_labels)
modelFilePath = os.path.join(models_path, modelFileName)
model = unetL5(pretrained_weights=modelFilePath,
               input_size=(256, 256, len(channels)),
               num_class=num_class)


def img2pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap



def getImgs(tilePath):
    imgs = list()
    i = 0
    if os.path.exists(tilePath):
        for channel in channels:
            imgs.append(io.imread(os.path.join(tilePath, channel + ".tif"), as_gray=True))
            i += 1
    return imgs


def updateVisualization():
    positionX = int(window.labelPositionX.text())
    positionY = int(window.labelPositionY.text())
    tile=str(positionX) + "\\" + str(positionY)

    tilePath = os.path.join(dataset_path, tile)
    if os.path.exists(tilePath):
        window.labelStatus.setText(tile)
        img_un = getImgs(tilePath)

        B2=img_un[1]
        B3=img_un[2]
        B4=img_un[3]
        B2 = (np.clip(B2, 0, 1500)) / 1500
        B3 = (np.clip(B3, 0, 1500)) / 1500
        B4 = (np.clip(B4, 0, 1500)) / 1500
        # B2 = (B2 - np.min(B2)) / np.max(B2)
        # B3 = (B3 - np.min(B3)) / np.max(B3)
        # B4 = (B4 - np.min(B4)) / np.max(B4)
        BGR = np.zeros((B2.shape[0], B2.shape[1], 3), np.uint8)
        BGR[:, :, 0] = B2 * 255
        BGR[:, :, 1] = B3 * 255
        BGR[:, :, 2] = B4 * 255
        window.labelFrameB4B3B2.setPixmap(img2pixmap(BGR))

        B5=img_un[4]
        B6=img_un[5]
        B7=img_un[6]
        # B5 = (np.clip(B5, 0, 1500)) / 1500
        # B6 = (np.clip(B6, 0, 1500)) / 1500
        # B7 = (np.clip(B7, 0, 1500)) / 1500
        B5 = (B5 - np.min(B5)) / np.max(B5)
        B6 = (B6 - np.min(B6)) / np.max(B6)
        B7 = (B7 - np.min(B7)) / np.max(B7)
        BGR2 = np.zeros((B5.shape[0], B5.shape[1], 3), np.uint8)
        BGR2[:, :, 0] = B5 * 255
        BGR2[:, :, 1] = B6 * 255
        BGR2[:, :, 2] = B7 * 255
        window.labelFrameB5B6B7.setPixmap(img2pixmap(BGR2))

        B8=img_un[7]
        B8=(B8-np.min(B8))/np.max(B8)
        B8show = np.zeros((B8.shape[0], B8.shape[1], 3), np.uint8)
        B8show[:, :, 0] = B8*255
        B8show[:, :, 1] = B8*255
        B8show[:, :, 2] = B8*255
        window.labelFrameB8.setPixmap(img2pixmap(B8show))

        B8A = img_un[8]
        B8A = (B8A - np.min(B8A)) / np.max(B8A)
        B8Ashow = np.zeros((B8A.shape[0], B8A.shape[1], 3), np.uint8)
        B8Ashow[:, :, 0] = B8A * 255
        B8Ashow[:, :, 1] = B8A * 255
        B8Ashow[:, :, 2] = B8A * 255
        window.labelFrameB8A.setPixmap(img2pixmap(B8Ashow))

        B1 = img_un[0]
        B9 = img_un[9]
        B10 = img_un[10]
        # B5 = (np.clip(B5, 0, 1500)) / 1500
        # B6 = (np.clip(B6, 0, 1500)) / 1500
        # B7 = (np.clip(B7, 0, 1500)) / 1500
        B1 = (B1 - np.min(B1)) / np.max(B1)
        B9 = (B9 - np.min(B9)) / np.max(B9)
        B10 = (B10 - np.min(B10)) / np.max(B10)
        BGR3 = np.zeros((B1.shape[0], B1.shape[1], 3), np.uint8)
        BGR3[:, :, 0] = B1 * 255
        BGR3[:, :, 1] = B9 * 255
        BGR3[:, :, 2] = B10 * 255
        window.labelFrameB1B9B10.setPixmap(img2pixmap(BGR3))

        B11 = img_un[11]
        B12 = img_un[12]
        B11 = (B11 - np.min(B11)) / np.max(B11)
        B12 = (B12 - np.min(B12)) / np.max(B12)
        BGR4 = np.zeros((B1.shape[0], B1.shape[1], 3), np.uint8)
        BGR4[:, :, 0] = B11 * 255
        BGR4[:, :, 1] = B12 * 255
        BGR4[:, :, 2] = np.ones(B1.shape)*0
        window.labelFrameB11B12.setPixmap(img2pixmap(BGR4))

        cos_gt = io.imread(os.path.join(tilePath, "COSV1.tif"), as_gray=True)
        cos_gt, _ = classAgregateCOS(cos_gt)
        cos_gt = labelVisualizeCOS(num_class, class_aggregation_COLOR_DICT, cos_gt)
        cos_gt = cos_gt[..., ::-1].copy()
        window.labelFrameGT.setPixmap(img2pixmap(cos_gt))

        imgs = data_getImgs(dataset_path, tile, dataStats)
        img = np.zeros((256,256) + (len(channels),))
        for i in range(len(imgs)):
            img[:, :, i] = imgs[i]

        img = np.array([img])
        cos_predict = model.predict(img)
        cos_predict = np.argmax(cos_predict[0], axis=-1)
        cos_predict = cos_predict.astype(np.uint8)
        cos_predict = labelVisualizeCOS(num_class, class_aggregation_COLOR_DICT, cos_predict)
        cos_predict = cos_predict[..., ::-1].copy()
        window.labelFramePredict.setPixmap(img2pixmap(cos_predict))

    else:
        window.labelStatus.setText("No Data for tile " + tile)
    import math
    def num2deg(xtile, ytile, zoom):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)

    (lat_deg, lon_deg) = num2deg(positionX+0.5, positionY+0.5, 14)

    url="https://www.openstreetmap.org/#map=14/" + str(lat_deg) + "/" + str(lon_deg)
    webView.load(QUrl(url))
    window.labelLatitude.setText(str(np.round(lat_deg, 9))+"º")
    window.labelLongitude.setText(str(np.round(lon_deg, 9))+"º")


def on_botaoUp_clicked():
    positionY=int(window.labelPositionY.text())-1
    window.labelPositionY.setText(str(positionY))
    updateVisualization()


def on_botaoDown_clicked():
    positionY=int(window.labelPositionY.text())+1
    window.labelPositionY.setText(str(positionY))
    updateVisualization()


def on_botaoLeft_clicked():
    positionX=int(window.labelPositionX.text())-1
    window.labelPositionX.setText(str(positionX))
    updateVisualization()


def on_botaoRight_clicked():
    positionX=int(window.labelPositionX.text())+1
    window.labelPositionX.setText(str(positionX))
    updateVisualization()


def initPosition():
    ipcaX = 7799
    ipcaY = 6110
    positionX=7799
    window.labelPositionX.setText(str(positionX))
    positionY=6110
    window.labelPositionY.setText(str(positionY))


def on_botaoHome_clicked():
    # print(webView.url())
    initPosition()
    updateVisualization()


print("Qt version: {}".format(qVersion()))
app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("mainViewQt.ui")

initPosition()
window.botaoUp.clicked.connect(on_botaoUp_clicked)
window.botaoDown.clicked.connect(on_botaoDown_clicked)
window.botaoLeft.clicked.connect(on_botaoLeft_clicked)
window.botaoRight.clicked.connect(on_botaoRight_clicked)
window.botaoHome.clicked.connect(on_botaoHome_clicked)

window.labelFrameB4B3B2.setScaledContents(True)
window.labelFrameB8.setScaledContents(True)
window.labelFrameB8A.setScaledContents(True)
window.labelFrameB11B12.setScaledContents(True)
window.labelFrameB5B6B7.setScaledContents(True)
window.labelFrameB1B9B10.setScaledContents(True)
window.labelFrameGT.setScaledContents(True)
window.labelFramePredict.setScaledContents(True)

vbox = QVBoxLayout(window.widgetMap)
webView = QWebEngineView()
vbox.addWidget(webView)
window.widgetMap.setLayout(vbox)
#webView.load(QUrl("https://www.google.pt/maps/@41.5333322,-8.6325896,14z"))
window.widgetMap.show()

on_botaoHome_clicked()

window.show()
sys.exit(app.exec())

