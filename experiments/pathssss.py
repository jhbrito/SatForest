import os

datasetPath = "C:\\Users\DiogoNunes\Documents\Tesselo\data\\tesselo-training-tiles"
with open('pathssss.txt', 'a') as filessss:
    tileXList = os.scandir(datasetPath)
    for tileX in tileXList:
        tileYList = os.scandir(tileX)
        filessss.write('\n' + os.path.join(tileX.name))
        for tileY in tileYList:
            filessss.write('\n\t' + os.path.join(tileY.name))