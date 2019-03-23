import numpy as np
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import os
import sys


from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

import tensorflow as tf




def switch_result_file(argument):
    switcher = {
        0: 'results_no_dropout',
        1: 'result_standard_dropout',
        2: 'result_maxpool_dropout',
        3: 'result_both'
    }
    return switcher.get(argument, 'Invalid')


def getAndResizeMask(train_ids, test_ids, C):
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), C.IMG_HEIGHT, C.IMG_WIDTH, C.IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), C.IMG_HEIGHT, C.IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = C.TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:C.IMG_CHANNELS]
        img = resize(img, (C.IMG_HEIGHT, C.IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((C.IMG_HEIGHT, C.IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (C.IMG_HEIGHT, C.IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    # Get and resize test images
    X_test = np.zeros((len(test_ids), C.IMG_HEIGHT, C.IMG_WIDTH, C.IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = C.TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:C.IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (C.IMG_HEIGHT, C.IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img

    return X_train, Y_train, X_test



def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.8, 1.0, 0.5):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def model_instance(C):
    # Build U-Net model
    if K.image_data_format() == 'channels_first':
        input_shape = (C.IMG_CHANNELS, C.IMG_HEIGHT, C.IMG_WIDTH)
    else:
        input_shape = (C.IMG_HEIGHT, C.IMG_WIDTH, C.IMG_CHANNELS)

    inputs = Input(shape=input_shape)


    s = Lambda(lambda x: x / 255) (inputs)


    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(C.standard_dropout) (c1) if C.enable_standard_dropout else c1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = Dropout(C.max_pool_maxpool_dropout) (c1) if C.enable_maxpool_dropout else c1
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(C.standard_dropout) (c2) if C.enable_standard_dropout else c2
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = Dropout(C.max_pool_maxpool_dropout) (c2) if C.enable_maxpool_dropout else c2
    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(C.standard_dropout) (c3) if C.enable_standard_dropout else c3
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = Dropout(C.max_pool_maxpool_dropout) (c3) if C.enable_maxpool_dropout else c3
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(C.standard_dropout) (c4) if C.enable_standard_dropout else c4
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    c4 = Dropout(C.max_pool_maxpool_dropout) (c4) if C.enable_maxpool_dropout else c4
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(C.standard_dropout) (c5) if C.enable_standard_dropout else c5
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = Dropout(C.max_pool_maxpool_dropout) (c5) if C.enable_maxpool_dropout else c5


    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(C.standard_dropout) (c6) if C.enable_standard_dropout else c6
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    c6 = Dropout(C.max_pool_maxpool_dropout) (c6) if C.enable_maxpool_dropout else c6

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(C.standard_dropout) (c7) if C.enable_standard_dropout else c7
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    c7 = Dropout(C.max_pool_maxpool_dropout) (c7) if C.enable_maxpool_dropout else c7

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(C.standard_dropout) (c8) if C.enable_standard_dropout else c8
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
    c8 = Dropout(C.max_pool_maxpool_dropout) (c8) if C.enable_maxpool_dropout else c8


    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(C.standard_dropout) (c9) if C.enable_standard_dropout else c9
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

    return model
