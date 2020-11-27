import numpy as np
from scipy.stats import entropy
from sklearn import preprocessing

# import cynthonized_functions_sig #cythonized function

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

from keras.backend.tensorflow_backend import set_session, get_session, clear_session

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
from DataFrameIteratorWithFilePath import *
import random

#custom design cython functions
import cynthonized_functions


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


def get_generator_with_filename(dataframe, C):
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()

    image_generator = DataFrameIteratorWithFilePath(dataframe, image_data_generator=image_datagen, x_col=0, y_col="class", target_size=(C.IMG_WIDTH, C.IMG_HEIGHT), batch_size=C.batch_size,  class_mode=None, seed= C.randomSeed)
    mask_generator = DataFrameIteratorWithFilePath(dataframe, image_data_generator=mask_datagen, target_size=(C.IMG_WIDTH, C.IMG_HEIGHT), x_col=1, batch_size=C.batch_size, color_mode='grayscale', class_mode=None, seed= C.randomSeed)


    generator = zip(image_generator, mask_generator)
    for d in generator:
        yield [*adjustData(d[0][0],d[1][0], C.num_classes), d[0][1], d[1][1]]

def get_generator(dataframe, C):
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()

    image_generator  = image_datagen.flow_from_dataframe(dataframe, target_size=(C.IMG_WIDTH, C.IMG_HEIGHT), x_col=0, batch_size=C.batch_size,  class_mode=None, seed= C.randomSeed)
    mask_generator = image_datagen.flow_from_dataframe(dataframe, target_size=(C.IMG_WIDTH, C.IMG_HEIGHT), x_col=1, batch_size=C.batch_size, color_mode='grayscale', class_mode=None, seed= C.randomSeed)

    generator = zip(image_generator, mask_generator)
    for d in generator:
        yield adjustData(
            d[0],d[1], C.num_classes)

def adjustData(img,mask,num_class):
    img = img / 255
    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]

    if (num_class) <= 2:
        mask /= 255
        return img, to_categorical(mask.astype(np.uint8), 2) # possible range of pixel values
    else:
        return img,to_categorical(mask.astype(np.uint8), num_class)



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

def get_uncertain_segmentation_image(seg_arr, n_classes ,colors=class_colors ):
    print (seg_arr[20,20,:])
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

def score_unlabeled_images(acquisition_type, generator_range, unlabeled_generator, network, C):
    #acquisition types
    unlabeled_scores_dict = dict()
    if acquisition_type == 0:
        # fully partial posterior query by dropout committee with KL
        print('Calculting Random Acquisition')
        for it in range(generator_range):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            for fn in fn_imgs:
                unlabeled_scores_dict[fn] = 1 # assign uniforma score to unlabeled set
            if len(unlabeled_scores_dict) > C.active_batch:
                break #stop when we have acquired enough for active batch
        unlabeled_scores_dict = sorted(unlabeled_scores_dict.items(), key=lambda kv: kv[1])
        print('Done Random Acquisition')

    elif acquisition_type == 1:
        # fully partial posterior query by dropout committee with KL
        print('Calculting entropy Acquisition')
        for it in range(generator_range):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = network.getModel().layers[-1].output.get_shape().as_list()[1:]
            stochastic_predictions = np.zeros((img.shape[0], *out_shape))
            for dropout_i in range(C.dropout_iterations):
                stochastic_predictions += network.stochastic_foward_pass(img)
            stochastic_predictions /= C.dropout_iterations
            scores = cynthonized_functions.score_entropy(stochastic_predictions)
            for score_index, s in enumerate(scores):
                unlabeled_scores_dict[fn_imgs[score_index]] = s
        unlabeled_scores_dict = sorted(unlabeled_scores_dict.items(), key=lambda kv: kv[1])
        print('Done calculting entropy Acquisition')
    elif acquisition_type == 2:
        print ("Calculating Bald Acquisition")
        for it in range(generator_range):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = network.getModel().layers[-1].output.get_shape().as_list()[1:]
            standard_prediction = network.getModel().predict(img)
            standard_entropy_scores = cynthonized_functions.score_entropy(standard_prediction.astype(np.double))
            stochastic_entropy_scores = np.zeros((img.shape[0]))
            for dropout_i in range(C.dropout_iterations):
                stochastic_entropy_scores += cynthonized_functions.score_entropy(network.stochastic_foward_pass(img).astype(np.double))
            stochastic_entropy_scores /= C.dropout_iterations
            bald_score = standard_entropy_scores - stochastic_entropy_scores
            for score_index, s in enumerate(bald_score):
                unlabeled_scores_dict[fn_imgs[score_index]] = s
        unlabeled_scores_dict = sorted(unlabeled_scores_dict.items(), key=lambda kv: kv[1])
        print('Done Bald Acquisition')
    elif acquisition_type == 3:
        print ("Calculating Committee-KL  Acquisition")
        for it in range(generator_range):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = network.getModel().layers[-1].output.get_shape().as_list()[1:]
            standard_prediction = network.getModel().predict(img).astype(np.double)
            stochastic_predictions = np.zeros((img.shape[0], *out_shape))
            for dropout_i in range(C.dropout_iterations):
                stochastic_predictions += network.stochastic_foward_pass(img)
            stochastic_predictions /= C.dropout_iterations #average stochastic prediction
            kl_score = cynthonized_functions.score_kl(standard_prediction, stochastic_predictions)
            for score_index, s in enumerate(kl_score):
                unlabeled_scores_dict[fn_imgs[score_index]] = s
        unlabeled_scores_dict = sorted(unlabeled_scores_dict.items(), key=lambda kv: kv[1])
        print('Done Calculating Commitee-KL Acquisition')

    elif acquisition_type == 4:
        print ("Calculating Committee-Jensen  Acquisition")
        for it in range(generator_range):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = network.getModel().layers[-1].output.get_shape().as_list()[1:]
            standard_prediction = network.getModel().predict(img).astype(np.double)
            stochastic_predictions = np.zeros((img.shape[0], *out_shape))
            for dropout_i in range(C.dropout_iterations):
                stochastic_predictions += network.stochastic_foward_pass(img)
            stochastic_predictions /= C.dropout_iterations #average stochastic prediction
            kl_score = cynthonized_functions.score_jensen(standard_prediction, stochastic_predictions)
            for score_index, s in enumerate(kl_score):
                unlabeled_scores_dict[fn_imgs[score_index]] = s
        unlabeled_scores_dict = sorted(unlabeled_scores_dict.items(), key=lambda kv: kv[1])
        print('Done Calculating Committee-Jensen')
    elif acquisition_type == 5:
        print ("Heteroscedastic Aleatoric uncertainty")
        for it in range(generator_range):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = network.getModel().layers[-1].output.get_shape().as_list()[1:]
            standard_prediction = network.getModel().predict(img).astype(np.double)
            stochastic_predictions = np.zeros((img.shape[0], *out_shape))
            stochastic_predictions_squared = np.zeros((img.shape[0], *out_shape))
            var_pred = np.zeros((img.shape[0], *out_shape[:-1], 1))
            print (stochastic_predictions.shape, stochastic_predictions_squared.shape, var_pred.shape)
            for dropout_i in range(C.dropout_iterations):
                pred = network.stochastic_foward_pass(img)
                stochastic_predictions_squared += pred **2
                stochastic_predictions += pred
                var_pred[:,:,:,0] += np.var(pred, axis = 3) ** 2
            var_pred /= C.dropout_iterations
            stochastic_predictions /= C.dropout_iterations #average stochastic prediction
            stochastic_predictions_squared /= C.dropout_iterations
            var_pred /= C.dropout_iterations
            stochastic_predictions = (stochastic_predictions) **2
            predictions = stochastic_predictions_squared - stochastic_predictions + var_pred
            scores = cynthonized_functions.score_entropy(predictions.astype(np.double))
            for score_index, s in enumerate(kl_score):
                unlabeled_scores_dict[fn_imgs[score_index]] = s
        unlabeled_scores_dict = sorted(unlabeled_scores_dict.items(), key=lambda kv: kv[1])

        print('Done Calculating Heteroscedastic Aleatoric')

    elif acquisition_type == 6:
        print ("MCMC Last Layer ")
        freeze = K.function([network.getModel().layers[0].input, K.learning_phase()],[network.getModel().layers[-2].output])
        last_layer_predict = K.function([network.getModel().layers[-2].output, K.learning_phase()],[network.getModel().layers[-1].output])
        for it in range(generator_range):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = network.getModel().layers[-1].output.get_shape().as_list()[1:]
            stochastic_predictions = np.zeros((img.shape[0], *out_shape))
            for dropout_i in range(C.dropout_iterations):
                intermidiate = freeze((img, 0))[0]
                pred = last_layer_predict((intermidiate, 1))[0]
                stochastic_predictions += pred
            stochastic_predictions /= C.dropout_iterations
            scores = cynthonized_functions.score_entropy(stochastic_predictions)
            for score_index, s in enumerate(scores):
                unlabeled_scores_dict[fn_imgs[score_index]] = s
        unlabeled_scores_dict = sorted(unlabeled_scores_dict.items(), key=lambda kv: kv[1])
        print('Done MCMC Last Layer')





    return unlabeled_scores_dict




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
