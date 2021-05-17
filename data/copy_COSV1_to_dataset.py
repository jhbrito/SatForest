import os
from shutil import copyfile

dataset_path = "C:/Tesselo/data/tesselo-training-tiles"
COS_path = "D:/data/COS/COS2015_v1/tiles"

tileXList = os.scandir(dataset_path)
for tileX in tileXList:
    tileYList = os.scandir(tileX)
    for tileY in tileYList:
        src_tile_COS_file = "cos2015-" + tileX.name + "-" + tileY.name + ".tif"
        print(src_tile_COS_file)
        src_tile_COS_path = os.path.join(COS_path,src_tile_COS_file)
        dst_tile_COS_path = os.path.join(dataset_path, tileX.name, tileY.name, "COSV1.tif")
        copyfile(src_tile_COS_path, dst_tile_COS_path)
