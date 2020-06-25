# options
import numpy as np
# from raster_classes import class_aggregation_8, class_aggregation_8_labels, class_aggregation_8_names
# from raster_classes import class_aggregation_5, class_aggregation_5_labels, class_aggregation_5_names
# from raster_classes import class_aggregation_4, class_aggregation_4_labels, class_aggregation_4_names, class_aggregation_4_COLOR_DICT
# from raster_classes import class_aggregation_2, class_aggregation_2_labels, class_aggregation_2_names, class_aggregation_2_COLOR_DICT
from raster_classes import class_aggregation_Eu3, class_aggregation_Eu3_labels, class_aggregation_Eu3_names, class_aggregation_Eu3_COLOR_DICT

dataset_path = "C:/Tesselo/data/tesselo-training-tiles"
results_path = "./results"

all_channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

use_max = False
level_i = 2  # default 2(5)
channels_i = 1  # default 1(64)
padding_i = 0  # default 0 (same)
batch_normalization_i = 0  # default 0 (None)
use_transpose_convolution_i = 0  # default 0 (False)

trainSize = -1  # -1 for all
testSize = -1  # -1 for all

epochs = 100
patience = 40

ignoreNODATA_flag = True
keepNODATA = False
# if not ignoreNODATA_flag and keepNODATA:
#     class_labels = [0, 1, 2, 3, 4, 9]
# else:
#     class_labels = np.arange(10).tolist()  # <--------------
class_aggregation = class_aggregation_Eu3
class_labels = class_aggregation_Eu3_labels
class_aggregation_names = class_aggregation_Eu3_names
class_aggregation_COLOR_DICT = class_aggregation_Eu3_COLOR_DICT

export_COS_files = True

# unet_models = [2, 3, 3.5, 4, 4.5]
unet_levels = (3, 4, 5, 6)  # original unet is 5 levels
net_channels_options = (32, 64, 128, 256)
padding_options = ('same', 'valid')
batch_normalization_options = ('None', 'CBNA', 'CABN', 'CBNADown', 'CABNDown')  # [No BN, BN between convolution and activation, BN after activation, BN between convolution and activation on the downward path, BN after activation on the downward path
use_transpose_convolution_options = [False, True]
batch_sizes = np.zeros((len(unet_levels),
                        len(net_channels_options),
                        len(padding_options),
                        len(batch_normalization_options),
                        len(use_transpose_convolution_options)))
# these are maximized for my GeForce GTX 1050, 4GB VRAM
batch_sizes[1, 0, 1, 1, 1] = 16  # level 4, 32,  valid, CBNA, TC
batch_sizes[2, 0, 1, 1, 1] = 22  # level 5, 32,  valid, CBNA, TC
batch_sizes[2, 1, 1, 0, 0] = 6  # level 5, 64,  valid, None, NO_TC

batch_sizes[1, 0, 0, 1, 1] = 9  # level 4, 32,  same, CBNA, TC
batch_sizes[1, 1, 0, 1, 0] = 4  # level 4, 64,  same, CBNA, NO_TC
batch_sizes[1, 1, 0, 2, 0] = 4  # level 4, 64,  same, CABN, NO_TC
batch_sizes[1, 1, 0, 1, 0] = 4  # level 4, 64,  same, CBNA, NO_TC  ## dropout 0.0

batch_sizes[1, 1, 0, 2, 0] = 4  # level 4, 64,  same, CABN, NO_TC, ## dropout 0.0
batch_sizes[1, 1, 0, 3, 0] = 4  # level 4, 64,  same, CBNADown, NO_TC ## dropout 0.0
batch_sizes[1, 1, 0, 4, 0] = 4  # level 4, 64,  same, CABNDown, NO_TC, ## dropout 0.0
batch_sizes[1, 1, 0, 3, 0] = 4  # level 4, 64,  same, CBNADown, NO_TC ## dropout 0.5
batch_sizes[1, 1, 0, 4, 0] = 4  # level 4, 64,  same, CABNDown, NO_TC, ## dropout 0.5
batch_sizes[1, 1, 0, 3, 0] = 4  # level 4, 64,  same, CBNADown, NO_TC ## dropout 0.25
batch_sizes[1, 1, 0, 4, 0] = 4  # level 4, 64,  same, CABNDown, NO_TC, ## dropout 0.25

batch_sizes[3, 0, 0, 0, 0] = 10  # level 6, 32,  same, None, NO_TC

batch_sizes[0, 1, 0, 0, 0] = 7  # level 3, 64,  same, None, NO_TC
batch_sizes[1, 1, 0, 0, 0] = 6  # level 4, 64,  same, None, NO_TC
batch_sizes[2, 1, 0, 0, 0] = 5  # level 5, 64,  same, None, NO_TC
batch_sizes[3, 1, 0, 0, 0] = 0  # level 6, 64,  same, None, NO_TC - impossível, nao há memória suficiente
batch_sizes[3, 1, 0, 0, 1] = 0  # level 6, 64,  same, None, TC - impossível, nao há memória suficiente

batch_sizes[2, 2, 0, 0, 0] = 0  # level 5, 128, same, None, NO_TC - impossível, nao há memória suficiente
batch_sizes[2, 2, 0, 0, 1] = 0  # level 5, 128, same, None, TC - impossível, nao há memória suficiente
batch_sizes[1, 2, 0, 0, 0] = 2  # level 4, 128, same, None, NO_TC

batch_sizes[1, 3, 0, 0, 1] = 0  # level 4, 256, same, None, TC - impossível, nao há memória suficiente
batch_sizes[1, 3, 0, 0, 0] = 0  # level 4, 256, same, None, NO_TC - impossível, nao há memória suficiente

batch_sizes[0, 0, 0, 0, 0] = 15  # level 3, 32,  same, None, NO_TC
batch_sizes[1, 0, 0, 0, 0] = 14  # level 4, 32,  same, None, NO_TC
batch_sizes[2, 0, 0, 0, 0] = 13  # level 5, 32,  same, None, NO_TC

# falta testar

batch_sizes[0, 2, 0, 0, 0] = 3  # level 3, 128, same, None, NO_TC
batch_sizes[0, 3, 0, 0, 0] = 1  # level 3, 256, same, None, NO_TC


def experiment_parameters(level_i, channels_i, padding_i, batch_normalization_i, use_transpose_convolution_i):
    unet_level = unet_levels[level_i]
    net_channels = net_channels_options[channels_i]
    padding = padding_options[padding_i]
    batch_normalization = batch_normalization_options[batch_normalization_i]
    use_transpose_convolution = use_transpose_convolution_options[use_transpose_convolution_i]
    dropout = 0.5  # 0.5
    batch_size = batch_sizes[level_i][channels_i][padding_i][batch_normalization_i][use_transpose_convolution_i]

    save_header = ('\nchannels: ' + str(channels) +
                   '\nLevel: ' + str(unet_level) +
                   '\nnet_starting_channels: ' + str(net_channels) +
                   '\npadding: ' + padding +
                   '\nbatch_normalization: ' + str(batch_normalization) +
                   '\ntranspose_convolution: ' + str(use_transpose_convolution) +
                   '\ndropout: ' + str(dropout) +
                   '\nbatch_size: ' + str(batch_size) +
                   '\n')

    return unet_level, net_channels, padding, batch_normalization, use_transpose_convolution, dropout, batch_size, save_header


unet_level, net_channels, padding, batch_normalization, use_transpose_convolution, dropout, batch_size, save_header = experiment_parameters(
    level_i=level_i, channels_i=channels_i, padding_i=padding_i, batch_normalization_i=batch_normalization_i,
    use_transpose_convolution_i=use_transpose_convolution_i)

# optionally override batch_size
batch_size = 4
