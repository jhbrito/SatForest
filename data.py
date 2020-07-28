# from __future__ import print_function
# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
# import glob
import skimage.io as io
import skimage.transform as trans
import random as r
# import matplotlib.pyplot as plt
import pickle
import rasterio as rio
from raster_classesV1 import class_lookup
from COS_train_options import class_aggregation, class_labels, class_aggregation_COLOR_DICT, channels, all_channels
from color_dictionary import Black


# Função que determina os parâmetros de treino
def parameters(ni):
    use_max = False
    do_batch_normalization = True
    use_transpose_convolution = True
    net_channels = 32
    drop = 0.5
    save_header = ('\nchannels: ' + str(channels) + '\nreshape: NO'
                                                    '\nbatch_size: 20'
                                                    '\nnormalization: MEAN_STD'
                                                    '\nbatch_normalization: YES'
                                                    '\ntranspose_convolution: YES'
                                                    '\nignoreNODATA: YES'
                                                    '\nnet_starting_channels: 32'
                                                    '\ndropout: 0.5\n')

    return use_max, do_batch_normalization, use_transpose_convolution, net_channels, drop, save_header


def moving_average(acc, n=3):
    ret = np.cumsum(acc, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Função para normalização do valor dos ficheiros
def normalizeB(imgB, channel, stats, use_max=False):
    if use_max:
        imgB = imgB / stats['max'][channel]
    else:
        imgB = (imgB - stats['mean'][channel]) / stats['std'][channel]
    return imgB


# Função que guarda os valores de normalização
def updateStatsB(stats, folder, channel):
    imgB = io.imread(os.path.join(folder, channel + ".tif"), as_gray=True)
    stats['max'][channel] = max(stats['max'][channel], imgB.max())
    stats['min'][channel] = min(stats['min'][channel], imgB.min())
    stats['mean'][channel].append(np.mean(imgB))
    stats['std'][channel].append(np.std(imgB))
    return stats


# Funçaõ que serve de loop para a normalização
def updateStats(stats, folder):
    for channel in all_channels:
        stats = updateStatsB(stats, folder, channel)

    return stats


# Funçaõ que efetua a preparação do dataset
def prepare_dataset(datasetPath, ignoreNODATAtiles=True, keepNODATA=False, cleanTilesFile='clean_tiles.txt',
                    dataStatsFile='data_stats.txt'):
    if os.path.isfile(cleanTilesFile) and os.path.isfile(dataStatsFile):
        # load tile list
        with open(cleanTilesFile, "rb") as fp:
            print("Loading tile list")
            paths = pickle.load(fp)
        with open(dataStatsFile, "rb") as fp:
            print("Loading data statistics")
            stats = pickle.load(fp)
    else:
        print('Preparing dataset...')
        stats = dict()
        stats['max'] = dict()
        stats['min'] = dict()
        stats['mean'] = dict()
        stats['std'] = dict()
        for channel in all_channels:
            stats['max'][channel] = np.iinfo(np.uint16).min
            stats['min'][channel] = np.iinfo(np.uint16).max
            stats['mean'][channel] = []
            stats['std'][channel] = []

        tileXList = os.scandir(datasetPath)
        paths = []
        i = 0
        for tileX in tileXList:
            tileYList = os.scandir(tileX)
            for tileY in tileYList:

                if ignoreNODATAtiles:
                    imgCOS_GT = io.imread(os.path.join(datasetPath, tileX.name, tileY.name, "COS.tif"), as_gray=True)
                    _, incompleteCOS = classAgregateCOS(imgCOS_GT)
                    print("image: ", i, "incomplete: ", incompleteCOS)
                    i += 1
                    if not incompleteCOS:
                        paths.append(os.path.join(tileX.name, tileY.name))
                        stats = updateStats(stats=stats, folder=os.path.join(datasetPath, tileX.name, tileY.name))
                else:
                    paths.append(os.path.join(tileX.name, tileY.name))
                    stats = updateStats(stats=stats, folder=os.path.join(datasetPath, tileX.name, tileY.name))
        # save tile list
        with open(cleanTilesFile, "wb") as fp:  # Pickling
            pickle.dump(paths, fp)
        # consolidate statistics
        # list of positive integer numbers
        for channel in all_channels:
            stats['mean'][channel] = np.mean(stats['mean'][channel])
            stats['std'][channel] = np.mean(stats['std'][channel])

        # save data statistics
        with open(dataStatsFile, "wb") as fp:  # Pickling
            pickle.dump(stats, fp)

    print("Splitting dataset")
    r.seed(1)
    total_imgs = len(paths)
    k = int(total_imgs * 0.8)
    trainSet = r.sample(paths, k)
    testSet = []
    for path in paths:
        if path not in trainSet:
            testSet.append(path)

    return trainSet, testSet, stats


# Funçaõ que efetua a preparação do dataset
def prepare_dataset_V1(datasetPath, cleanTilesFile='clean_paths_V1.txt', dataStatsFile='data_stats_V1.txt'):
    if os.path.isfile(cleanTilesFile):
        # load tile list
        with open(cleanTilesFile, "rb") as fp:
            print("Loading tile list")
            paths = pickle.load(fp)
    else:
        print('Preparing dataset...')
        tileXList = os.scandir(datasetPath)
        paths = []
        i = 0
        for tileX in tileXList:
            tileYList = os.scandir(tileX)
            for tileY in tileYList:

                imgCOS = io.imread(os.path.join(datasetPath, tileX.name, tileY.name, "COSV1.tif"), as_gray=True)
                nodataImg = np.zeros(imgCOS.shape, dtype='uint8')
                nodataImg[imgCOS == 99] = 1
                if nodataImg.sum() > 0:
                    incompleteData = True
                else:
                    incompleteData = False
                oceanImg = np.zeros(imgCOS.shape, dtype='uint8')
                oceanImg[imgCOS == 47] = 1
                if oceanImg.sum() >= imgCOS.shape[0]*imgCOS.shape[1]:
                    all_ocean=True
                else:
                    all_ocean = False
                print("image: ", i, "; incomplete: ", incompleteData, "; all ocean:", all_ocean)
                i += 1
                if not incompleteData and not all_ocean:
                    paths.append(os.path.join(tileX.name, tileY.name))
        # save tile list
        with open(cleanTilesFile, "wb") as fp:  # Pickling
            pickle.dump(paths, fp)

    print("Splitting dataset")
    r.seed(1)
    total_imgs = len(paths)
    k = int(total_imgs * 0.8)
    trainSet = r.sample(paths, k)
    testSet = []
    for path in paths:
        if path not in trainSet:
            testSet.append(path)

    # compute statistics on training set
    if os.path.isfile(dataStatsFile):
        with open(dataStatsFile, "rb") as fp:
            print("Loading training data statistics")
            stats = pickle.load(fp)
    else:
        print("Computing training data statistics")
        stats = dict()
        stats['max'] = dict()
        stats['min'] = dict()
        stats['mean'] = dict()
        stats['std'] = dict()
        for channel in all_channels:
            stats['max'][channel] = np.iinfo(np.uint16).min
            stats['min'][channel] = np.iinfo(np.uint16).max
            stats['mean'][channel] = []
            stats['std'][channel] = []

        i=1
        for path in trainSet:
            if (i % 100) == 0:
                print(str(i), "/", len(trainSet))
            stats = updateStats(stats=stats, folder=os.path.join(datasetPath, path))
            i=i+1
        # consolidate statistics
        for channel in all_channels:
            stats['mean'][channel] = np.mean(stats['mean'][channel])
            stats['std'][channel] = np.mean(stats['std'][channel])
        # save data statistics
        with open(dataStatsFile, "wb") as fp:  # Pickling
            pickle.dump(stats, fp)

    return trainSet, testSet, stats

# Função para agregação da máscara da COS, conforme os parâmetros do presente treino
def classAgregateCOS(imgCOS):
    nodataImg = np.zeros(imgCOS.shape, dtype='uint8')
    nodataImg[imgCOS == 99] = 1
    if nodataImg.sum() > 0:
        incompleteData = True
    else:
        incompleteData = False

    newImgCOS = np.zeros(imgCOS.shape, dtype='uint8')

    for classe in range(len(class_lookup) - 1):
        # classeLabel = class_lookup[classe]
        # newclasse = int(classeLabel[0]) - 1
        '''
        # For more than 5 classes
        if classe <= 12:
            newclasse = int(classeLabel[0]) - 1
        elif classe >= 42:
            newclasse = 30
        else:
            newclasse = classe - 12
        '''
        # Semi-final class agregation
        newclasse = class_aggregation[classe]  # newclasse = class_mask[classe]

        newImgCOS[imgCOS == classe] = newclasse
    # Keep NODTA or transform into ocean
    # newclasse = 9 if keepNODATA else 4
    # newImgCOS[imgCOS == 99] = newclasse
    # --->
    # newImgCOS[newImgCOS == 4] = 3 # Aggregate humids with water zones

    return newImgCOS, incompleteData


# Função que transforma a máscara da COS de uma matriz 256x256x1 para uma matriz 256x256xnum_classes
def normalizeCOS(imgCOS, num_class=5):
    newImgCOS, incompleteCOS = classAgregateCOS(imgCOS)
    new_mask = np.zeros(newImgCOS.shape + (num_class,))
    for i in range(num_class):
        new_mask[newImgCOS == i, i] = 1.

    return new_mask, incompleteCOS


# Função para leitura do ficheiro de imagem e normalização da mesma
def getImgs(datasetPath, tile, dataStats, use_max=False):
    imgs = list()
    i = 0
    for channel in channels:
        imgs.append(io.imread(os.path.join(datasetPath, tile, channel + ".tif"), as_gray=True))
        imgs[i] = normalizeB(imgs[i], channel, dataStats, use_max)
        i += 1

    return imgs


# Função que efetua o recorte da imagem, conforme os parâmetros de treino
def do_center_crop(img, new_size):
    cropy = (int)((img.shape[0] - new_size[0]) / 2)
    cropx = (int)((img.shape[1] - new_size[1]) / 2)
    img = img[cropy:img.shape[0] - cropy, cropx:img.shape[1] - cropx]
    return img


# Função para alteração das imagens de treino
def augmentImages(aug_dict, imgs, input_size, imgCOS, target_size):
    if 'width_shift_range' in aug_dict:
        input_cropx = r.sample(aug_dict['width_shift_range'], 1)[0]
    else:
        input_cropx = (int)((imgs[0].shape[1] - input_size[1]) / 2)
    if 'height_shift_range' in aug_dict:
        input_cropy = r.sample(aug_dict['height_shift_range'], 1)[0]
    else:
        input_cropy = (int)((imgs[0].shape[0] - input_size[0]) / 2)
    if 'rotation_range' in aug_dict:
        rotation = r.sample(aug_dict['rotation_range'], 1)[0]
    else:
        rotation = 0
    if 'horizontal_flip' in aug_dict and aug_dict['horizontal_flip']:
        do_horizontal_flip = r.sample([False, True], 1)[0]
    else:
        do_horizontal_flip = False
    if 'vertical_flip' in aug_dict and aug_dict['vertical_flip']:
        do_vertical_flip = r.sample([False, True], 1)[0]
    else:
        do_vertical_flip = False

    target_offsety = int((input_size[0] - target_size[0]) / 2)
    target_offsetx = int((input_size[1] - target_size[1]) / 2)
    if imgCOS.shape[0] != target_size[0] | imgCOS.shape[1] != target_size[1]:  # if use_unet == 5:
        imgCOS = imgCOS[target_offsety + input_cropy:target_offsety + input_cropy + target_size[0],
                 target_offsetx + input_cropx:target_offsetx + input_cropx + target_size[1]]

    if rotation:
        imgCOS = trans.rotate(imgCOS, rotation)
    if do_horizontal_flip:
        imgCOS = imgCOS[:, ::-1]
    if do_vertical_flip:
        imgCOS = imgCOS[::-1, :]

    for i in range(len(imgs)):
        img = imgs[i]
        if img.shape[0] != input_size[0] | img.shape[1] != input_size[1]:  # if use_unet == 5:
            img = img[input_cropy:input_cropy + input_size[0], input_cropx:input_cropx + input_size[1]]
        if rotation:
            img = trans.rotate(img, rotation)
        if do_horizontal_flip:
            img = img[:, ::-1]
        if do_vertical_flip:
            img = img[::-1, :]
        imgs[i] = img
    return imgs, imgCOS


# Função para gerar e gerir os dados de treino que são alimentados à rede neuronal
def trainGeneratorCOS(batch_size, datasetPath, trainSet, dataStats, aug_dict, input_size=(256, 256),
                      target_size=(256, 256), num_classes=5, use_max=False, ignoreNODATA_flag=True):
    if batch_size > 1:
        while 1:
            iTile = 0
            nBatches = int(np.ceil(len(trainSet) / batch_size))
            for batchID in range(nBatches):
                # img_un = np.zeros((256, 256, len(channels)), dtype = np.float)
                imgs = np.zeros(((batch_size,) + input_size + (len(channels),)))
                imgsCOS = np.zeros(((batch_size,) + target_size + (num_classes,)))
                iTileInBatch = 0
                while iTileInBatch < batch_size:
                    if iTile < len(trainSet):
                        tile = trainSet[iTile]
                        iTile += 1
                        imgCOS = io.imread(os.path.join(datasetPath, tile, "COSV1.tif"), as_gray=True)
                        imgCOS, incompleteCOS = normalizeCOS(imgCOS, num_classes)
                        if not incompleteCOS or not ignoreNODATA_flag:
                            img_un = getImgs(datasetPath, tile, dataStats, use_max)
                            img_un, imgCOS = augmentImages(aug_dict, img_un, input_size, imgCOS, target_size)
                            imgsCOS[iTileInBatch, :, :, :] = imgCOS
                            for i in range(len(img_un)):
                                imgs[iTileInBatch, :, :, i] = img_un[i]
                            iTileInBatch += 1

                    else:
                        imgs = imgs[0:iTileInBatch, :, :, :]
                        imgsCOS = imgsCOS[0:iTileInBatch, :, :, :]
                        break

                yield (imgs, imgsCOS)
    else:
        while 1:
            for tile in trainSet:
                imgCOS = io.imread(os.path.join(datasetPath, tile, "COSV1.tif"), as_gray=True)
                imgCOS, incompleteCOS = normalizeCOS(imgCOS, num_classes)
                if incompleteCOS:
                    continue

                imgs = getImgs(datasetPath, tile, dataStats, use_max)
                imgs, imgCOS = augmentImages(aug_dict, imgs, input_size, imgCOS, target_size)
                img = np.zeros(input_size + (len(channels),))
                for i in range(len(imgs)):
                    img[:, :, i] = imgs[i]

                imgCOS = np.array([imgCOS])
                img = np.array([img])

                yield (img, imgCOS)


# Função para gerar e gerir os dados de teste que são alimentados à rede neuronal
def testGeneratorCOS(datasetPath, testSet, dataStats, input_size=(256, 256), use_max=False):
    for tile in testSet:
        imgs = getImgs(datasetPath, tile, dataStats, use_max)
        for i in range(len(channels)):
            imgs[i] = do_center_crop(imgs[i], input_size)

        img = np.zeros(input_size + (len(channels),))
        for i in range(len(imgs)):
            img[:, :, i] = imgs[i]

        img = np.array([img])
        yield (img)


# Função que transforma a máscar da COS numa máscara de cores, para ser visualmente distinta
def labelVisualizeCOS(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,), dtype='uint8')
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    img_out[img == 99, :] = Black
    return img_out


# Função para gravara os resultados de teste
def saveResultCOS(datasetPath, testSet, results, resultsPath, target_size, export_COS_files=False):
    num_class = results.shape[-1]
    y_gt = np.zeros(((len(testSet),) + target_size))
    y_predict = np.zeros(((len(testSet),) + target_size))
    if export_COS_files and not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
    # testpaths_file = open('C:\\Users\DiogoNunes\Documents\Tesselo\data\\test_set_paths.txt', 'w') # Save test paths
    for i, item in enumerate(results):
        filename = testSet[i]
        filename = filename.replace("\\", "_")

        # testpaths_file.write(filename)
        # testpaths_file.write('\n')

        cos_gt = rio.open(os.path.join(datasetPath, testSet[i], "COSV1.tif"))
        cos_meta = cos_gt.meta.copy()
        #### FIX META DATA
        cos_meta['height'] = target_size[0]
        cos_meta['width'] = target_size[1]
        cos_meta3 = cos_meta.copy()
        cos_meta3['count'] = 3

        cos_gt = cos_gt.read(1)
        cos_gt = do_center_crop(cos_gt, target_size)
        cos_gt, _ = classAgregateCOS(cos_gt)
        # out_cos_gt = rio.open(os.path.join(resultsPath, filename + "_COS_GT10.tif"), 'w', **cos_meta) # GT10
        # out_cos_gt.write(cos_gt, 1)
        # out_cos_gt.close()

        y_gt[i, ...] = cos_gt  # Save for confusion matrix

        if export_COS_files:
            cos_gt = labelVisualizeCOS(num_class, class_aggregation_COLOR_DICT, cos_gt)  # SEMI_COLOR_DICT
            out_cos_gt = rio.open(os.path.join(resultsPath, filename + "_COS_GT" + str(len(class_labels)) + "_colour.tif"), 'w',
                                  **cos_meta3)  # GT10_colour
            out_cos_gt.write(cos_gt[:, :, 0], 1)
            out_cos_gt.write(cos_gt[:, :, 1], 2)
            out_cos_gt.write(cos_gt[:, :, 2], 3)
            out_cos_gt.close()

        cos_predict = np.argmax(item, axis=-1)
        cos_predict = cos_predict.astype(np.uint8)
        # out_cos_predict = rio.open(os.path.join(resultsPath, filename + "_COS_predict10.tif"), 'w', **cos_meta) # predict10
        # out_cos_predict.write(cos_predict, 1)
        # out_cos_predict.close()

        y_predict[i, ...] = cos_predict  # Save for confusion matrix

        if export_COS_files:
            cos_predict = labelVisualizeCOS(num_class, class_aggregation_COLOR_DICT, cos_predict)  # SEMI_COLOR_DICT
            out_cos_predict = rio.open(os.path.join(resultsPath, filename + "_COS_predict" + str(len(class_labels)) + "_colour.tif"), 'w',
                                       **cos_meta3)  # predict10_colour
            out_cos_predict.write(cos_predict[:, :, 0], 1)
            out_cos_predict.write(cos_predict[:, :, 1], 2)
            out_cos_predict.write(cos_predict[:, :, 2], 3)
            out_cos_predict.close()

        if (i % 50) == 0:
            perc = (i / len(testSet)) * 100
            print('%.2f' % perc, '%')

    # testpaths_file.close()
    return y_gt, y_predict
