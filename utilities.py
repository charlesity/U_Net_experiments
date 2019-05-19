import numpy as np
from scipy.stats import entropy
from sklearn import preprocessing

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

from Network import *


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

def model_instance(C):
    net_instance = Network(C)
    return net_instance



def bald(net_instance, unlabeled_X, C):
    # print ("parsed unlabeled shape ",unlabeled_X.shape)
    shuffled_indices = np.arange(unlabeled_X.shape[0])
    np.random.shuffle(shuffled_indices)
    #take a subset of the unlabeled data
    subsampled_indices = shuffled_indices[:C.subsample_size]
    subsampled_unlabeled_X = unlabeled_X[subsampled_indices]

    #calculate the acquition function using only subsampled_unlabeled_X
    the_shape = np.append(subsampled_unlabeled_X.shape[0], C.output_shape)
    score_All = np.zeros(shape=(the_shape))
    All_Entropy_Dropout = np.zeros(shape=subsampled_unlabeled_X.shape[0])

    for d in range(C.dropout_iterations):
        dropout_score = net_instance.stochastic_foward_pass(subsampled_unlabeled_X)
        score_All = score_All + dropout_score
        Entropy_Per_Dropout = np.zeros((dropout_score.shape[0]))
        for i in range(dropout_score.shape[0]):
            Entropy_Per_Dropout[i] = entropy(dropout_score[i].flatten())
        All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout # sum across all pixel entropies

    Avg_Dropout_Entropy = np.divide(All_Entropy_Dropout, C.dropout_iterations)
    Avg_Score = np.divide(score_All, C.dropout_iterations)
    Avg_entropy = np.zeros((Avg_Score.shape[0]))

    for i in range(Avg_entropy.shape[0]):
        Avg_entropy[i] = entropy(Avg_Score[i].flatten())

    U_X = Avg_entropy - Avg_Dropout_Entropy
    arg_max = U_X.argsort()[-C.active_batch:][::-1] # max indices within subsampled_unlabeled_X

    #get the corresponding index the in unlabeled_X implicitly within subsampled_indices
    max_info_indices = subsampled_indices[arg_max] # their corresponding index values with the most information
    return max_info_indices

def independent_pixel_entropy(net_instance, unlabeled_X, C):
    # print ("parsed unlabeled shape ",unlabeled_X.shape)
    shuffled_indices = np.arange(unlabeled_X.shape[0])
    np.random.shuffle(shuffled_indices)
    #take a subset of the unlabeled data
    subsampled_indices = shuffled_indices[:C.subsample_size]
    subsampled_unlabeled_X = unlabeled_X[subsampled_indices]

    all_entropys = np.zeros(shape=(subsampled_unlabeled_X.shape[0]))
    prediction_distribution = np.zeros(shape=(subsampled_unlabeled_X.shape[0],subsampled_unlabeled_X.shape[1]*subsampled_unlabeled_X.shape[2], C.dropout_iterations))
    for d in range(C.dropout_iterations):
        dropout_score = net_instance.stochastic_foward_pass(subsampled_unlabeled_X).reshape(subsampled_unlabeled_X.shape[0]
                                                                                            ,subsampled_unlabeled_X.shape[1]*subsampled_unlabeled_X.shape[2])
        prediction_distribution[:,:,d] = dropout_score

    #calculate entropy per pixel predictions
    lm_entropy = lambda d: entropy(d)
    for i, images in enumerate(prediction_distribution):
        result = np.array(list(map(lm_entropy, images)))
        all_entropys[i] = result.sum() # sum of pixel entropies

    arg_max = all_entropys.argsort()[-C.active_batch:][::-1]
    #get the corresponding index the in unlabeled_X implicitly within subsampled_indices
    max_info_indices = subsampled_indices[arg_max] # their corresponding index values with the most information
    return max_info_indices
