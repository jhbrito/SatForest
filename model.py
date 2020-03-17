import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import metrics
from data import channels

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
    conv11 = Reshape([input_size[0] * input_size[1], num_class])(conv10)

    #model = Model(input = inputs, output = conv10)
    model = Model(input=inputs, output=conv11)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
'''


def unet_v2(pretrained_weights = None, input_size = (256, 256, len(channels)), num_class = 5):

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

    conv10 = Conv2D(num_class, 1, activation = 'softmax', kernel_initializer = 'he_normal')(conv9)
    #conv11 = Reshape([input_size[0] * input_size[1], num_class])(conv10)

    #model = Model(input=inputs, output=conv11)
    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


# unet_v3 has padding = 'same'
def unet_v3(pretrained_weights = None, input_size = (256, 256, len(channels)), num_class = 2,
            do_batch_normalization = False, use_transpose_convolution = True, net_channels = 64, dropout = 0.5):

    inputs = Input(input_size)

    conv1 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if do_batch_normalization else conv1
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if do_batch_normalization else conv1
    conv1 = Activation('relu')(conv1)

    #drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    net_channels *= 2

    conv2 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if do_batch_normalization else conv2
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if do_batch_normalization else conv2
    conv2 = Activation('relu')(conv2)

    #drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
    net_channels *= 2

    conv3 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if do_batch_normalization else conv3
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if do_batch_normalization else conv3
    conv3 = Activation('relu')(conv3)

    #drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
    net_channels *= 2

    conv4 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if do_batch_normalization else conv4
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if do_batch_normalization else conv4
    conv4 = Activation('relu')(conv4)

    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(drop4)
    net_channels *= 2

    conv5 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5) if do_batch_normalization else conv5
    conv5 = Activation('relu')(conv5)

    conv5 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5) if do_batch_normalization else conv5
    conv5 = Activation('relu')(conv5)

    drop5 = Dropout(dropout)(conv5)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up6 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(drop5)
    else:
        up6 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchNormalization()(up6) if do_batch_normalization else up6
    up6 = Activation('relu')(up6)

    merge6 = concatenate([drop4, up6], axis = 3)

    conv6 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6) if do_batch_normalization else conv6
    conv6 = Activation('relu')(conv6)

    conv6 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6) if do_batch_normalization else conv6
    conv6 = Activation('relu')(conv6)

    #drop6 = Dropout(dropout)(conv6)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up7 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(conv6)
    else:
        up7 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7) if do_batch_normalization else up7
    up7 = Activation('relu')(up7)

    merge7 = concatenate([conv3, up7], axis = 3)

    conv7 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if do_batch_normalization else conv7
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if do_batch_normalization else conv7
    conv7 = Activation('relu')(conv7)

    #drop7 = Dropout(dropout)(conv7)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up8 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(conv7)
    else:
        up8 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8) if do_batch_normalization else up8
    up8 = Activation('relu')(up8)

    merge8 = concatenate([conv2, up8], axis = 3)

    conv8 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if do_batch_normalization else conv8
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if do_batch_normalization else conv8
    conv8 = Activation('relu')(conv8)

    #drop8 = Dropout(dropout)(conv8)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up9 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(conv8)
    else:
        up9 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9) if do_batch_normalization else up9
    up9 = Activation('relu')(up9)

    merge9 = concatenate([conv1, up9], axis = 3)

    conv9 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if do_batch_normalization else conv9
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if do_batch_normalization else conv9
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(num_class, 1, activation = 'softmax', kernel_initializer = 'he_normal')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

# unet_v3B has padding = 'same' and one less level
def unet_v3B(pretrained_weights = None, input_size = (256, 256, len(channels)), num_class = 2,
            do_batch_normalization = False, use_transpose_convolution = True, net_channels = 64, dropout = 0.5):

    inputs = Input(input_size)

    conv1 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if do_batch_normalization else conv1
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if do_batch_normalization else conv1
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    net_channels *= 2

    conv2 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if do_batch_normalization else conv2
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if do_batch_normalization else conv2
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
    net_channels *= 2

    conv3 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if do_batch_normalization else conv3
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if do_batch_normalization else conv3
    conv3 = Activation('relu')(conv3)

    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(drop3)
    net_channels *= 2

    conv4 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if do_batch_normalization else conv4
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if do_batch_normalization else conv4
    conv4 = Activation('relu')(conv4)

    drop4 = Dropout(dropout)(conv4)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up7 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(drop4)
    else:
        up7 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
    up7 = BatchNormalization()(up7) if do_batch_normalization else up7
    up7 = Activation('relu')(up7)

    merge7 = concatenate([drop3, up7], axis = 3)

    conv7 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if do_batch_normalization else conv7
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if do_batch_normalization else conv7
    conv7 = Activation('relu')(conv7)

    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up8 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(conv7)
    else:
        up8 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8) if do_batch_normalization else up8
    up8 = Activation('relu')(up8)

    merge8 = concatenate([conv2, up8], axis = 3)

    conv8 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if do_batch_normalization else conv8
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if do_batch_normalization else conv8
    conv8 = Activation('relu')(conv8)

    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up9 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(conv8)
    else:
        up9 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9) if do_batch_normalization else up9
    up9 = Activation('relu')(up9)

    merge9 = concatenate([conv1, up9], axis = 3)

    conv9 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if do_batch_normalization else conv9
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(net_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if do_batch_normalization else conv9
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(num_class, 1, activation = 'softmax', kernel_initializer = 'he_normal')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

# unet_v4 has padding = 'valid'
def unet_v4(pretrained_weights = None, input_size = (252, 252, len(channels)), num_class = 2,
            do_batch_normalization = False, use_transpose_convolution = True, net_channels = 64, dropout = 0.5):

    inputs = Input(input_size)

    conv1 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if do_batch_normalization else conv1
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if do_batch_normalization else conv1
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    net_channels *= 2

    conv2 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if do_batch_normalization else conv2
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if do_batch_normalization else conv2
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
    net_channels *= 2

    conv3 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if do_batch_normalization else conv3
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if do_batch_normalization else conv3
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
    net_channels *= 2

    conv4 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if do_batch_normalization else conv4
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if do_batch_normalization else conv4
    conv4 = Activation('relu')(conv4)

    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(drop4)
    net_channels *= 2

    conv5 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5) if do_batch_normalization else conv5
    conv5 = Activation('relu')(conv5)

    conv5 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5) if do_batch_normalization else conv5
    conv5 = Activation('relu')(conv5)

    drop5 = Dropout(0.5)(conv5)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up6 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(drop5)
    else:
        up6 = Conv2D(net_channels, 2, padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchNormalization()(up6) if do_batch_normalization else up6
    up6 = Activation('relu')(up6)

    merge6 = concatenate([Cropping2D(cropping = ((4, 4), (4, 4)))(drop4),up6], axis = 3)

    conv6 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6) if do_batch_normalization else conv6
    conv6 = Activation('relu')(conv6)

    conv6 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6) if do_batch_normalization else conv6
    conv6 = Activation('relu')(conv6)

    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up7 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(conv6)
    else:
        up7 = Conv2D(net_channels, 2, padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7) if do_batch_normalization else up7
    up7 = Activation('relu')(up7)

    merge7 = concatenate([Cropping2D(cropping = ((16, 16), (16, 16)))(conv3),up7], axis = 3)

    conv7 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if do_batch_normalization else conv7
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if do_batch_normalization else conv7
    conv7 = Activation('relu')(conv7)

    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up8 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(conv7)
    else:
        up8 = Conv2D(net_channels, 2, padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8) if do_batch_normalization else up8
    up8 = Activation('relu')(up8)

    merge8 = concatenate([Cropping2D(cropping = ((40, 40), (40, 40)))(conv2),up8], axis = 3)

    conv8 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if do_batch_normalization else conv8
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if do_batch_normalization else conv8
    conv8 = Activation('relu')(conv8)

    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up9 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(conv8)
    else:
        up9 = Conv2D(net_channels, 2, padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9) if do_batch_normalization else up9
    up9 = Activation('relu')(up9)

    merge9 = concatenate([Cropping2D(cropping = ((88, 88), (88, 88)))(conv1),up9], axis = 3)

    conv9 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if do_batch_normalization else conv9
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if do_batch_normalization else conv9
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(num_class, 1, activation = 'softmax', kernel_initializer = 'he_normal')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

# unet_v5 has padding = 'valid' and one less level
def unet_v5(pretrained_weights = None, input_size = (252, 252, len(channels)), num_class = 5,
            do_batch_normalization = False, use_transpose_convolution = True, net_channels = 64, dropout = 0.5):

    inputs = Input(input_size)

    conv1 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if do_batch_normalization else conv1
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if do_batch_normalization else conv1
    conv1 = Activation('relu')(conv1)

    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(drop1)
    net_channels *= 2

    conv2 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if do_batch_normalization else conv2
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(net_channels , 3, padding = 'valid', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if do_batch_normalization else conv2
    conv2 = Activation('relu')(conv2)

    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    net_channels *= 2

    conv3 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if do_batch_normalization else conv3
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if do_batch_normalization else conv3
    conv3 = Activation('relu')(conv3)

    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    net_channels *= 2

    conv4 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if do_batch_normalization else conv4
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if do_batch_normalization else conv4
    conv4 = Activation('relu')(conv4)

    drop4 = Dropout(dropout)(conv4)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up7 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(drop4)
    else:
        up7 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
    up7 = BatchNormalization()(up7) if do_batch_normalization else up7
    up7 = Activation('relu')(up7)

    merge7 = concatenate([Cropping2D(cropping = ((4, 4), (4, 4)))(drop3), up7], axis = 3)

    conv7 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if do_batch_normalization else conv7
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if do_batch_normalization else conv7
    conv7 = Activation('relu')(conv7)

    drop7 = Dropout(dropout)(conv7)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up8 = Conv2DTranspose(net_channels * 2, (2, 2), strides = (2, 2))(drop7)
    else:
        up8 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
    up8 = BatchNormalization()(up8) if do_batch_normalization else up8
    up8 = Activation('relu')(up8)

    merge8 = concatenate([Cropping2D(cropping = ((16, 16), (16, 16)))(conv2),up8], axis = 3)

    conv8 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if do_batch_normalization else conv8
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if do_batch_normalization else conv8
    conv8 = Activation('relu')(conv8)

    drop8 = Dropout(dropout)(conv8)
    net_channels = int(net_channels / 2)

    if use_transpose_convolution:
        up9 = Conv2DTranspose(net_channels, (2, 2), strides = (2, 2))(drop8)
    else:
        up9 = Conv2D(net_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop8))
    up9 = BatchNormalization()(up9) if do_batch_normalization else up9
    up9 = Activation('relu')(up9)

    merge9 = concatenate([Cropping2D(cropping = ((40, 40), (40, 40)))(conv1), up9], axis = 3)

    conv9 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if do_batch_normalization else conv9
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(net_channels, 3, padding = 'valid', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if do_batch_normalization else conv9
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(num_class, 1, activation = 'softmax', kernel_initializer = 'he_normal')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model #, model_b

