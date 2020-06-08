# import os
# import skimage.io as io
# import skimage.transform as trans
# import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from data import channels

# unet_models = [2, 3, 3.5, 4, 4.5]
unet_levels = (3, 4, 5, 6)  # original unet is 5 levels
net_channels_options = (32, 64, 128, 256)
padding_options = ('same', 'valid')
input_valid_sizes = (252, 252, 252)
output_valid_sizes = (212, 164, 68)
batch_normalization_options = ('None', 'CBNA', 'CABN', 'CBNADown',
                               'CABNDown')  # [No BN, BN between convolution and activation, BN after activation, BN between convolution and activation on the downward path, BN after activation on the downward path
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

# falta testar

batch_sizes[2, 0, 0, 0, 0] = 13  # level 5, 32,  same, None, NO_TC

batch_sizes[0, 2, 0, 0, 0] = 3  # level 3, 128, same, None, NO_TC
batch_sizes[0, 3, 0, 0, 0] = 1  # level 3, 256, same, None, NO_TC


def experiment_parameters(level_i, channels_i, padding_i, batch_normalization_i, use_transpose_convolution_i):
    unet_level = unet_levels[level_i]
    net_channels = net_channels_options[channels_i]
    padding = padding_options[padding_i]
    batch_normalization = batch_normalization_options[batch_normalization_i]
    use_transpose_convolution = use_transpose_convolution_options[use_transpose_convolution_i]
    dropout = 0.5  # 0.5
    batch_size = 0  # batch_sizes[level_i][channels_i]

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


'''
def unet(pretrained_weights = None, input_size = (256, 256, 1), num_class = 2):

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(num_class, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    conv10 = Conv2D(num_class, 1, activation='softmax')(conv9)
    
    #model = Model(input = inputs, output = conv10)
    model = Model(input=inputs, output=conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
'''


# unet is the original unet with padding = 'same' and multiclass
def unet(pretrained_weights=None,
         input_size=(256, 256, len(channels)),
         net_channels=64,
         num_class=2):
    inputs = Input(input_size)

    conv1 = Conv2D(net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(2 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(4 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(8 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(16 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(16 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(8 * net_channels, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(8 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(8 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(4 * net_channels, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(4 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(4 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(2 * net_channels, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(2 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(2 * net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(net_channels, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(net_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(num_class, 1, activation='softmax', kernel_initializer='he_normal')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# unetL3 has 3 levels unet with options
def unetL3(pretrained_weights=None,
           input_size=(256, 256, len(channels)),
           num_class=2,
           net_channels=64,
           padding='same',  # or valid
           batch_normalization=None,  # 'None'/'CBNA'/'CABN'/'CBNADown'/'CABNDown'
           use_transpose_convolution=False,
           dropout=0.5):
    inputs = Input(input_size)

    conv1 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if (
                (batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv1
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv1

    conv1 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv1
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv1

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv2
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv2

    conv2 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv2
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv2

    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if batch_normalization == 'CBNA' else conv3
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3) if batch_normalization == 'CABN' else conv3

    conv3 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if batch_normalization == 'CBNA' else conv3
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3) if batch_normalization == 'CABN' else conv3

    drop3 = Dropout(dropout)(conv3)

    if use_transpose_convolution:
        up8 = Conv2DTranspose(2 * net_channels, (2, 2), strides=(2, 2))(drop3)
    else:
        up8 = Conv2D(2 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop3))
    up8 = BatchNormalization()(up8) if batch_normalization == 'CBNA' else up8
    up8 = Activation('relu')(up8)
    up8 = BatchNormalization()(up8) if batch_normalization == 'CABN' else up8

    if padding == 'valid':
        drop2 = Cropping2D(cropping=((4, 4), (4, 4)))(drop2)
    merge8 = concatenate([drop2, up8], axis=3)

    conv8 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CBNA' else conv8
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CABN' else conv8

    conv8 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CBNA' else conv8
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CABN' else conv8

    if use_transpose_convolution:
        up9 = Conv2DTranspose(net_channels, (2, 2), strides=(2, 2))(conv8)
    else:
        up9 = Conv2D(net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9) if batch_normalization == 'CBNA' else up9
    up9 = Activation('relu')(up9)
    up9 = BatchNormalization()(up9) if batch_normalization == 'CABN' else up9

    if padding == 'valid':
        conv1 = Cropping2D(cropping=((16, 16), (16, 16)))(conv1)
    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CBNA' else conv9
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CABN' else conv9

    conv9 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CBNA' else conv9
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CABN' else conv9

    conv10 = Conv2D(num_class, 1, activation='softmax', kernel_initializer='he_normal')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


# unetL4 has 4 levels unet with options
def unetL4(pretrained_weights=None,
           input_size=(256, 256, len(channels)),
           num_class=2,
           net_channels=64,
           padding='same',  # or valid
           batch_normalization=None,  # 'None'/'CBNA'/'CABN'/'CBNADown'/'CABNDown'
           use_transpose_convolution=False,
           dropout=0.5):
    inputs = Input(input_size)

    conv1 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv1
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv1

    conv1 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv1
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv1

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv2
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv2

    conv2 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv2
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv2

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv3
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv3

    conv3 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv3
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv3

    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if batch_normalization == 'CBNA' else conv4
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4) if batch_normalization == 'CABN' else conv4

    conv4 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if batch_normalization == 'CBNA' else conv4
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4) if batch_normalization == 'CABN' else conv4

    drop4 = Dropout(dropout)(conv4)

    if use_transpose_convolution:
        up7 = Conv2DTranspose(4 * net_channels, (2, 2), strides=(2, 2))(drop4)
    else:
        up7 = Conv2D(4 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
    up7 = BatchNormalization()(up7) if batch_normalization == 'CBNA' else up7
    up7 = Activation('relu')(up7)
    up7 = BatchNormalization()(up7) if batch_normalization == 'CABN' else up7

    if padding == 'valid':
        drop3 = Cropping2D(cropping=((4, 4), (4, 4)))(drop3)
    merge7 = concatenate([drop3, up7], axis=3)

    conv7 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CBNA' else conv7
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CABN' else conv7

    conv7 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CBNA' else conv7
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CABN' else conv7

    if use_transpose_convolution:
        up8 = Conv2DTranspose(2 * net_channels, (2, 2), strides=(2, 2))(conv7)
    else:
        up8 = Conv2D(2 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8) if batch_normalization == 'CBNA' else up8
    up8 = Activation('relu')(up8)
    up8 = BatchNormalization()(up8) if batch_normalization == 'CABN' else up8

    if padding == 'valid':
        conv2 = Cropping2D(cropping=((16, 16), (16, 16)))(conv2)
    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CBNA' else conv8
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CABN' else conv8

    conv8 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CBNA' else conv8
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CABN' else conv8

    if use_transpose_convolution:
        up9 = Conv2DTranspose(net_channels, (2, 2), strides=(2, 2))(conv8)
    else:
        up9 = Conv2D(net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9) if batch_normalization == 'CBNA' else up9
    up9 = Activation('relu')(up9)
    up9 = BatchNormalization()(up9) if batch_normalization == 'CABN' else up9

    if padding == 'valid':
        conv1 = Cropping2D(cropping=((40, 40), (40, 40)))(conv1)
    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CBNA' else conv9
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CABN' else conv9

    conv9 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CBNA' else conv9
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CABN' else conv9

    conv10 = Conv2D(num_class, 1, activation='softmax', kernel_initializer='he_normal')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


# unetL5 is the original unet with 5 levels with options
def unetL5(pretrained_weights=None,
           input_size=(256, 256, len(channels)),
           num_class=2,
           net_channels=64,
           padding='same',  # or valid
           batch_normalization=None,  # 'None'/'CBNA'/'CABN'/'CBNADown'/'CABNDown'
           use_transpose_convolution=False,
           dropout=0.5):
    inputs = Input(input_size)

    conv1 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv1
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv1

    conv1 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv1
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv1

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv2
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv2

    conv2 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv2
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv2

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv3
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv3

    conv3 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv3
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv3

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv4
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv4

    conv4 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv4
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv4

    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(16 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5) if batch_normalization == 'CBNA' else conv5
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5) if batch_normalization == 'CABN' else conv5

    conv5 = Conv2D(16 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5) if batch_normalization == 'CBNA' else conv5
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5) if batch_normalization == 'CABN' else conv5

    drop5 = Dropout(dropout)(conv5)

    if use_transpose_convolution:
        up6 = Conv2DTranspose(8 * net_channels, (2, 2), strides=(2, 2))(drop5)
    else:
        up6 = Conv2D(8 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    up6 = BatchNormalization()(up6) if batch_normalization == 'CBNA' else up6
    up6 = Activation('relu')(up6)
    up6 = BatchNormalization()(up6) if batch_normalization == 'CABN' else up6

    if padding == 'valid':
        drop4 = Cropping2D(cropping=((4, 4), (4, 4)))(drop4)

    merge6 = concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6) if batch_normalization == 'CBNA' else conv6
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6) if batch_normalization == 'CABN' else conv6

    conv6 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6) if batch_normalization == 'CBNA' else conv6
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6) if batch_normalization == 'CABN' else conv6

    if use_transpose_convolution:
        up7 = Conv2DTranspose(4 * net_channels, (2, 2), strides=(2, 2))(conv6)
    else:
        up7 = Conv2D(4 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))

    up7 = BatchNormalization()(up7) if batch_normalization == 'CBNA' else up7
    up7 = Activation('relu')(up7)
    up7 = BatchNormalization()(up7) if batch_normalization == 'CABN' else up7

    if padding == 'valid':
        conv3 = Cropping2D(cropping=((16, 16), (16, 16)))(conv3)
    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CBNA' else conv7
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CABN' else conv7

    conv7 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CBNA' else conv7
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CABN' else conv7

    if use_transpose_convolution:
        up8 = Conv2DTranspose(2 * net_channels, (2, 2), strides=(2, 2))(conv7)
    else:
        up8 = Conv2D(2 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8) if batch_normalization == 'CBNA' else up8
    up8 = Activation('relu')(up8)
    up8 = BatchNormalization()(up8) if batch_normalization == 'CABN' else up8

    if padding == 'valid':
        conv2 = Cropping2D(cropping=((40, 40), (40, 40)))(conv2)
    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CBNA' else conv8
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CABN' else conv8

    conv8 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CBNA' else conv8
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CABN' else conv8

    if use_transpose_convolution:
        up9 = Conv2DTranspose(net_channels, (2, 2), strides=(2, 2))(conv8)
    else:
        up9 = Conv2D(net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9) if batch_normalization == 'CBNA' else up9
    up9 = Activation('relu')(up9)
    up9 = BatchNormalization()(up9) if batch_normalization == 'CABN' else up9

    if padding == 'valid':
        conv1 = Cropping2D(cropping=((88, 88), (88, 88)))(conv1)
    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CBNA' else conv9
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CABN' else conv9

    conv9 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CBNA' else conv9
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CABN' else conv9

    conv10 = Conv2D(num_class, 1, activation='softmax', kernel_initializer='he_normal')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


# unetL6 is unet with 6 levels with options
def unetL6(pretrained_weights=None,
           input_size=(256, 256, len(channels)),
           num_class=2,
           net_channels=64,
           batch_normalization=None,  # 'None'/'CBNA'/'CABN'/'CBNADown'/'CABNDown'
           use_transpose_convolution=False,
           dropout=0.5):
    inputs = Input(input_size)

    padding = 'same'

    conv1 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv1
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv1

    conv1 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv1
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv1

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv2
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv2

    conv2 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv2
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv2

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv3
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv3

    conv3 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv3
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv3

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv4
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv4

    conv4 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv4
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv4

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(16 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv5
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv5

    conv5 = Conv2D(16 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5) if ((batch_normalization == 'CBNA') | (batch_normalization == 'CBNADown')) else conv5
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5) if ((batch_normalization == 'CABN') | (batch_normalization == 'CABNDown')) else conv5

    drop5 = Dropout(dropout)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)

    conv5B = Conv2D(32 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(pool5)
    conv5B = BatchNormalization()(conv5B) if batch_normalization == 'CBNA' else conv5B
    conv5B = Activation('relu')(conv5B)
    conv5B = BatchNormalization()(conv5B) if batch_normalization == 'CABN' else conv5B

    conv5B = Conv2D(32 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv5B)
    conv5B = BatchNormalization()(conv5B) if batch_normalization == 'CBNA' else conv5B
    conv5B = Activation('relu')(conv5B)
    conv5B = BatchNormalization()(conv5B) if batch_normalization == 'CABN' else conv5B

    drop5B = Dropout(dropout)(conv5B)

    if use_transpose_convolution:
        up5C = Conv2DTranspose(16 * net_channels, (2, 2), strides=(2, 2))(drop5B)
    else:
        up5C = Conv2D(16 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5B))
    up5C = BatchNormalization()(up5C) if batch_normalization == 'CBNA' else up5C
    up5C = Activation('relu')(up5C)
    up5C = BatchNormalization()(up5C) if batch_normalization == 'CABN' else up5C

    merge5C = concatenate([drop5, up5C], axis=3)

    conv5C = Conv2D(16 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge5C)
    conv5C = BatchNormalization()(conv5C) if batch_normalization == 'CBNA' else conv5C
    conv5C = Activation('relu')(conv5C)
    conv5C = BatchNormalization()(conv5C) if batch_normalization == 'CABN' else conv5C

    conv5C = Conv2D(16 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv5C)
    conv5C = BatchNormalization()(conv5C) if batch_normalization == 'CBNA' else conv5C
    conv5C = Activation('relu')(conv5C)
    conv5C = BatchNormalization()(conv5C) if batch_normalization == 'CABN' else conv5C

    if use_transpose_convolution:
        up6 = Conv2DTranspose(8 * net_channels, (2, 2), strides=(2, 2))(conv5C)
    else:
        up6 = Conv2D(8 * net_channels, 2, padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv5C))
    up6 = BatchNormalization()(up6) if batch_normalization == 'CBNA' else up6
    up6 = Activation('relu')(up6)
    up6 = BatchNormalization()(up6) if batch_normalization == 'CABN' else up6

    merge6 = concatenate([conv4, up6], axis=3)

    conv6 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6) if batch_normalization == 'CBNA' else conv6
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6) if batch_normalization == 'CABN' else conv6

    conv6 = Conv2D(8 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6) if batch_normalization == 'CBNA' else conv6
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6) if batch_normalization == 'CABN' else conv6

    if use_transpose_convolution:
        up7 = Conv2DTranspose(4 * net_channels, (2, 2), strides=(2, 2))(conv6)
    else:
        up7 = Conv2D(4 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))

    up7 = BatchNormalization()(up7) if batch_normalization == 'CBNA' else up7
    up7 = Activation('relu')(up7)
    up7 = BatchNormalization()(up7) if batch_normalization == 'CABN' else up7

    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CBNA' else conv7
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CABN' else conv7

    conv7 = Conv2D(4 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CBNA' else conv7
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization == 'CABN' else conv7

    if use_transpose_convolution:
        up8 = Conv2DTranspose(2 * net_channels, (2, 2), strides=(2, 2))(conv7)
    else:
        up8 = Conv2D(2 * net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8) if batch_normalization == 'CBNA' else up8
    up8 = Activation('relu')(up8)
    up8 = BatchNormalization()(up8) if batch_normalization == 'CABN' else up8

    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CBNA' else conv8
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CABN' else conv8

    conv8 = Conv2D(2 * net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CBNA' else conv8
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization == 'CABN' else conv8

    if use_transpose_convolution:
        up9 = Conv2DTranspose(net_channels, (2, 2), strides=(2, 2))(conv8)
    else:
        up9 = Conv2D(net_channels, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9) if batch_normalization == 'CBNA' else up9
    up9 = Activation('relu')(up9)
    up9 = BatchNormalization()(up9) if batch_normalization == 'CABN' else up9

    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CBNA' else conv9
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CABN' else conv9

    conv9 = Conv2D(net_channels, 3, padding=padding, kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CBNA' else conv9
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization == 'CABN' else conv9

    conv10 = Conv2D(num_class, 1, activation='softmax', kernel_initializer='he_normal')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model
