import numpy as np
from scipy.stats import entropy
from sklearn import preprocessing

import cynthonized_functions_sig #cythonized function

from tqdm import tqdm
from itertools import chain
# from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import matplotlib.pyplot as plt
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
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
from keras.utils import to_categorical
K.clear_session()

import tensorflow as tf

from Network import *
import random


class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


def switch_result_file(argument):
    switcher = {
        0: 'results_no_dropout',
        1: 'result_standard_dropout',
        2: 'result_maxpool_dropout',
        3: 'result_both'
    }
    return switcher.get(argument, 'Invalid')

def get_generator(data_gen_args, dataframe, C, sample_images):
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(sample_images[0],augment=True, seed= C.randomSeed)
    mask_datagen.fit(sample_images[1],augment=True, seed= C.randomSeed)

    image_generator  = image_datagen.flow_from_dataframe(dataframe, target_size=(C.IMG_WIDTH, C.IMG_HEIGHT), x_col=0, batch_size=C.batch_size,  class_mode=None, seed= C.randomSeed)
    mask_generator = image_datagen.flow_from_dataframe(dataframe, target_size=(C.IMG_WIDTH, C.IMG_HEIGHT), x_col=1, batch_size=C.batch_size, class_mode=None, seed= C.randomSeed)

    generator = zip(image_generator, mask_generator)
    flag_multi_class = True if C.num_classes > 2 else False
    for d in generator:
        yield adjustData(
            d[0],d[1], flag_multi_class, C.num_classes)

def adjustData(img,mask,flag_multi_class,num_class):
    img = img / 255
    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
    mask = mask /255
    return img, to_categorical(mask, num_class)


def get_colored_segmentation_image(seg_arr, n_classes ,colors=class_colors ):
    seg_arr = seg_arr.argmax(axis=2)
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_img[:, :, 0] += (seg_arr[:, :] == c)*(colors[c][0])
        seg_img[:, :, 1] += (seg_arr[:, :] == c)*(colors[c][1])
        seg_img[:, :, 2] += (seg_arr[:, :] == c)*(colors[c][2])

    return seg_img.astype(np.uint8)


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

    # all_entropys = np.zeros(shape=(subsampled_unlabeled_X.shape[0]))
    # prediction_distribution = np.zeros(shape=(len(subsampled_unlabeled_X), *C.output_shape, C.dropout_iterations))
    prediction_distribution = np.zeros(shape=(len(subsampled_unlabeled_X), subsampled_unlabeled_X.shape[1]*subsampled_unlabeled_X.shape[2], C.dropout_iterations))

    for inter in range(C.dropout_iterations):
        dropout_score = net_instance.stochastic_foward_pass(subsampled_unlabeled_X).reshape(subsampled_unlabeled_X.shape[0]
                                                                                            ,subsampled_unlabeled_X.shape[1]*subsampled_unlabeled_X.shape[2])
        # dropout_score = net_instance.stochastic_foward_pass(subsampled_unlabeled_X)
        # prediction_distribution[:,:,:,:, inter] = dropout_score
        prediction_distribution[:,:, inter] = dropout_score

    # prediction_distribution = prediction_distribution.mean(axis = 4)
    all_entropys = cynthonized_functions_sig.score_entropy_sig(prediction_distribution)
    arg_max = all_entropys.argsort()[-C.active_batch:][::-1]
    #get the corresponding index in unlabeled_X implicitly within subsampled_indices
    max_info_indices = subsampled_indices[arg_max] # their corresponding index values with the most information
    return max_info_indices

def independent_pixel_entropy_generator(net_instance, unlabeled_X, C, frame_location, mask_location, image_shape, output_shape):

    # print ("parsed unlabeled shape ",unlabeled_X.shape)
    shuffled_indices = np.arange(unlabeled_X.shape[0])
    np.random.shuffle(shuffled_indices)
    #take a subset of the unlabeled data
    subsampled_indices = shuffled_indices[:C.subsample_size]
    subsampled_unlabeled_X = unlabeled_X[subsampled_indices]

    sub_sample_gen = data_gen(frame_location, mask_location, image_shape, output_shape, subsampled_unlabeled_X, subsampled_unlabeled_X, batch_size = len(subsampled_unlabeled_X))
    subSampledData_X, _ = next(sub_sample_gen)
    # all_entropys = np.zeros(shape=(subsampled_unlabeled_X.shape[0]))
    prediction_distribution = np.zeros(shape=(subSampledData_X.shape[0],subSampledData_X.shape[1]*subSampledData_X.shape[2], C.dropout_iterations))
    for d in range(C.dropout_iterations):
        dropout_score = net_instance.stochastic_foward_pass(subSampledData_X).reshape(subSampledData_X.shape[0]
                                                                                            ,subSampledData_X.shape[1]*subSampledData_X.shape[2])
        prediction_distribution[:,:,d] = dropout_score


    all_entropys = cynthonized_functions_sig.score_entropy_sig(prediction_distribution)
    arg_max = all_entropys.argsort()[-C.active_batch:][::-1]
    #get the corresponding index in unlabeled_X implicitly within subsampled_indices
    max_info_indices = subsampled_indices[arg_max] # their corresponding index values with the most information
    return max_info_indices


def mean_iou(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.ravel()
    y_pred_f = y_pred.ravel()

    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f*y_true_f) + np.sum(y_pred_f*y_pred_f) + smooth)

def data_gen(frame_location, mask_location, image_shape, output_shape, imgs, masks, batch_size):
    c = 0
    np.random.shuffle(imgs) #shuffle training images
    while (True):
        img = np.zeros((batch_size, *image_shape)).astype('float')
        mask = np.zeros((batch_size, *output_shape)).astype('float')
        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0.
            train_img = cv2.imread(frame_location+'/'+imgs[i], -1)/255.
            train_img =  cv2.resize(train_img, image_shape[:2])# Read an image from folder and resize
            # img[i-c] = np.expand_dims(train_img, axis = 2) #add to array - img[0], img[1], and so on.

            train_mask = cv2.imread(mask_location+'/'+imgs[i][:-4]+"_anno.bmp", cv2.IMREAD_GRAYSCALE)/255.
            train_mask = cv2.resize(train_mask, output_shape[:2])
            train_mask[train_mask > 0] = 1
            train_mask = train_mask.reshape(output_shape) # Add extra dimension for parity with train_img size [512 * 512 * 3]
            mask[i-c] = train_mask
        c+=batch_size
        if(c+batch_size>=len(imgs)):
            c=0
            np.random.shuffle(imgs)  #  "randomizing again"
        yield img, mask
