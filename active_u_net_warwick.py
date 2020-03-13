#Code written by Charles I. Saidu https://github.com/charlesity
#All rights reserved
import math
import numpy as np
import argparse
import sys
import cv2

from Config import Config
from keras.datasets import mnist
from keras import backend as K
K.clear_session()
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import time
import random
import argparse

from utilities import *
from Network import *
from sklearn.metrics import jaccard_score # for mean_iou

from keras.utils import plot_model

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""



#
start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("-ds", "--dataset_location",  required = True, help ="link to the datasource")
parser.add_argument("-qt", "--q_type", type=int,
                    help="1 = bald acquition, else = random acquition")
parser.add_argument("-re_w", "--re_initialize_weights", type=int,
                    help="determine whether or not weights should be re-initialized")
parser.add_argument("-mc_p", "--mc_prediction", type=int,
                help="Make prediction via MC")
parser.add_argument("-early_s", "--early_stop", type=int,
                    help="Indicates early stop")
parser.add_argument("-reg", "--regularizers", type=int,
                    help="Weight regularizers")



def get_training_test_set(frame_location, mask_location, split_size):
    all_frames = os.listdir(frame_location)
    all_masks = os.listdir(mask_location)
    random.shuffle(all_frames)

    train_split = int(split_size*len(all_frames))

    train_frames = all_frames[:train_split]
    test_frames = all_frames[train_split:]
    # Generate corresponding mask lists for masks
    train_masks = [f for f in all_masks if f[:-9]+".bmp" in train_frames]
    test_masks = [f for f in all_masks if f[:-9]+".bmp" in test_frames]


    return np.array(train_frames), np.array(train_masks), np.array(test_frames), np.array(test_masks)

#
image_shape = (128, 128, 3)
output_shape = (128,128,1)
arg = parser.parse_args()
C = Config()
C.set_enable_dropout(2)  # max_pool dropout type
C.regularizers = bool(arg.regularizers)

random.seed = C.seed
np.random.seed = C.seed
dataset_location = arg.dataset_location
C.early_stop = arg.early_stop

frame_location = dataset_location +"images"
mask_location = dataset_location +"masks"


C.IMG_HEIGHT, C.IMG_WIDTH, C.IMG_CHANNELS = image_shape


# directory to save result
script_file = os.path.splitext(os.path.basename(__file__))[0]
if not os.path.exists('./results/'+script_file):
    results_location = './results/'+script_file
    os.mkdir(results_location)
else:
    results_location = './results/'+script_file

test_results_mean_iou = np.zeros((C.n_experiments, C.num_active_queries+1))
results_mean_iou = np.zeros((C.n_experiments, C.num_active_queries+1))  #in-sample mean_iou
#
num_samples = []

for e in range(C.n_experiments):
    C.active_batch = 15

    gen_batch_size = 4
    train_frames, train_masks, test_frames, test_masks = get_training_test_set(frame_location, mask_location, 0.7)
    num_training_images = len(train_frames)


    all_indices = np.arange(len(train_frames))

    np.random.shuffle(all_indices)
    ac_ind = int(C.initial_training_ratio * len(train_frames))   #

    print (ac_ind)

    # print (len(train_frames), len(test_frames), len(train_masks), len(test_masks),ac_ind)
    # exit()
    if ac_ind < gen_batch_size:
        ac_ind = gen_batch_size

    active_indices = all_indices[:ac_ind].copy()


    un_labeled_index = all_indices[ac_ind:].copy()


    # use the training set for active learning
    active_train_X = train_frames[active_indices]
    active_train_y = train_masks[active_indices]


    unlabeled_X = train_frames[un_labeled_index]
    unlabeled_y = train_masks[un_labeled_index]

    C.output_shape = output_shape

    train_gen = data_gen(frame_location, mask_location, image_shape, output_shape, active_train_X, active_train_y, batch_size = gen_batch_size)
    test_gen = data_gen(frame_location, mask_location, image_shape,output_shape, test_frames,test_masks, batch_size = len(test_frames)) # fetch all at once


    net_instance = model_instance(C)
    X_test, Y_test = next(test_gen)


    if C.early_stop == 1:
        es = EarlyStopping(monitor='dice_coef', mode='auto',patience=C.early_stop_patience, verbose=1)
        net_instance.getModel().fit_generator(train_gen, epochs=C.epoch, callbacks=[es], steps_per_epoch = (num_training_images//gen_batch_size))
    else:
        net_instance.getModel().fit_generator(train_gen, epochs=C.epoch, steps_per_epoch = (num_training_images//gen_batch_size))


    num_active_training_images = len(active_train_X)
    total_steps = num_active_training_images // gen_batch_size if num_active_training_images % gen_batch_size == 0 else (num_active_training_images // gen_batch_size) + 1

    if arg.mc_prediction == 0:
        pred_mask = net_instance.getModel().predict_generator(train_gen, steps = total_steps)  #training sample prediction
        test_pred_mask = net_instance.getModel().predict(X_test)
    else:
        pred_mask, _ , active_train_y_mask = net_instance.stochastic_predict_generator(train_gen, total_steps, C)
        test_pred_mask, _ = net_instance.stochastic_predict(X_test, C)

    train_mean_iou = mean_iou(active_train_y_mask[:num_active_training_images], pred_mask[:num_active_training_images])
    test_mean_iou = mean_iou(Y_test, test_pred_mask)

    results_mean_iou[e][0] = train_mean_iou
    test_results_mean_iou[e][0] = test_mean_iou


    print (train_mean_iou, test_mean_iou)


    if e == 0:
        num_samples.append(num_active_training_images)

    #retrieve weights in case weight re-initialization is set to 1
    current_weights = net_instance.getModel().get_weights()

    for aquisitions in range(C.num_active_queries):
        #use all the unlabeled dataset as subsample size
        C.subsample_size = len(unlabeled_X)
        print (aquisitions+1, " active acquition ", " experiment num ", e+1)
        if arg.q_type == 1:  # fully partial posterior query by dropout committee with KL
            print('bald acquition')
            index_maximum=bald(net_instance, unlabeled_X, C)
        elif arg.q_type == 2: #random
            print ('Random acquisition')
            shuffled_indices = np.arange(unlabeled_X.shape[0])
            np.random.shuffle(shuffled_indices)
            index_maximum = shuffled_indices[:C.active_batch]
        elif arg.q_type == 3:
            print ('Entropy with independent pixel assumptions')
            index_maximum = independent_pixel_entropy_generator(net_instance, unlabeled_X, C, frame_location, mask_location, image_shape, output_shape)
        else:
            exit()

        additional_samples_X = unlabeled_X[index_maximum].copy() # most informative samples
        additional_samples_y = unlabeled_y[index_maximum].copy() # get their corresponding mask


        # print ("Before delete Number of unlabeled x and y", unlabeled_X.shape, unlabeled_y.shape)
        # print ("Before delete Number of active x and y", active_train_X.shape, active_train_y.shape)

        unlabeled_X = np.delete(unlabeled_X, index_maximum, axis=0)
        unlabeled_y = np.delete(unlabeled_y, index_maximum, axis=0)


        active_train_X = np.append(active_train_X, additional_samples_X, axis=0)
        active_train_y = np.append(active_train_y, additional_samples_y, axis=0)


        # print ("After delete Number of unlabeled x and y", unlabeled_X.shape, unlabeled_y.shape)
        # print ("After delete Number of active x and y", active_train_X.shape, active_train_y.shape)

        # # uncomment to visualized 9 informative unlabeled and informative samples
        # fig = plt.figure(figsize=(4, 25))
        # idx = 0
        # num_rows = 5
        # for i in range(num_rows):
        #     idx +=1
        #     ax = fig.add_subplot(num_rows, 2, idx, xticks=[], yticks=[])
        #     ax.imshow(additional_samples_X[i, :, :, :], cmap='Greys')
        #     idx +=1
        #     ax = fig.add_subplot(num_rows, 2, idx, xticks=[], yticks=[])
        #     mask = net_instance.getModel().predict(np.expand_dims(additional_samples_X[i, :, :, :], axis=0))
        #     print (mask.shape)
        #     # exit()
        #     ax.imshow(mask[0, :, :, 0], cmap='Greys')
        # fig.suptitle("Most informative samples")
        # plt.show()
        #
        if arg.re_initialize_weights == 0:
            net_instance.getModel().set_weights(current_weights)  #weight fine-tuning
        else:
            net_instance = model_instance(C)  #re-initialize weights


        train_gen = data_gen(frame_location, mask_location, image_shape,output_shape, active_train_X, active_train_y, batch_size = gen_batch_size)


        if C.early_stop == 1:
            es = EarlyStopping(monitor='dice_coef', mode='auto',patience=C.early_stop_patience, verbose=1)
            net_instance.getModel().fit_generator(train_gen, epochs=C.epoch, callbacks=[es], steps_per_epoch = (num_training_images//gen_batch_size))
        else:
            net_instance.getModel().fit_generator(train_gen, epochs=C.epoch, steps_per_epoch = (num_training_images//gen_batch_size))


        num_active_training_images = len(active_train_X)
        total_steps = num_active_training_images // gen_batch_size if num_active_training_images % gen_batch_size == 0 else (num_active_training_images // gen_batch_size) + 1

        if arg.mc_prediction == 0:
            pred_mask = net_instance.getModel().predict_generator(train_gen, steps = total_steps)  #training sample prediction
            test_pred_mask = net_instance.getModel().predict(X_test)
        else:
            pred_mask, _ , active_train_y_mask = net_instance.stochastic_predict_generator(train_gen, total_steps, C)
            test_pred_mask, _ = net_instance.stochastic_predict(X_test, C)

        # # uncomment for iterative visualization
        # fig = plt.figure(figsize=(4, 25))
        # idx = 0
        # num_rows = 5
        # for i in np.arange(num_rows):
        #     idx+=1
        #     ax = fig.add_subplot(num_rows, 3, idx, xticks=[], yticks=[])
        #     ax.imshow(X_test_with_mask[i, :, :, :])
        #     ax.set_title('Actual Test')
        #     idx+=1
        #     ax = fig.add_subplot(num_rows, 3, idx, xticks=[], yticks=[])
        #     ax.imshow(y_test_mask[i, :, :, 0], cmap='Greys')
        #     ax.set_title('Mask Test')
        #     idx+=1
        #     ax = fig.add_subplot(num_rows, 3, idx, xticks=[], yticks=[])
        #     p = test_pred_mask[i, :, :, 0]
        #     p[p >= C.mask_threshold] = 1
        #     p[p < C.mask_threshold] = 0  # threshold predicted mask
        #     ax.imshow(p, cmap='Greys')
        #     ax.set_title('Predicted Mask Test')
        # plt.show()

        train_mean_iou = mean_iou(active_train_y_mask[:num_active_training_images], pred_mask[:num_active_training_images])
        test_mean_iou = mean_iou(Y_test, test_pred_mask)


        print (train_mean_iou, test_mean_iou)


        results_mean_iou[e][aquisitions+1] = train_mean_iou
        test_results_mean_iou[e][aquisitions+1] = test_mean_iou

        if e == 0:
            num_samples.append(active_train_X.shape[0])

        # print (results_mean_iou[e][:(aquisitions+2)])
        #
        # ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
        # ax.plot(results_mean_iou[e][:(aquisitions+2)])
        # ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
        # ax.plot(test_results_mean_iou[e][:(aquisitions+2)])
        #
        # plt.draw()


results_mean_iou = results_mean_iou.mean(axis = 0)
test_results_mean_iou = test_results_mean_iou.mean(axis = 0)

result_file = script_file+"_train_"+str(arg.q_type)+"_"+str(arg.re_initialize_weights)+"_"+str(arg.mc_prediction)+"_"+ str(C.early_stop)+"_"+str(C.regularizers)+"_.npy"
test_result_file = script_file+"_test_"+str(arg.q_type)+"_"+str(arg.re_initialize_weights)+"_"+str(arg.mc_prediction)+"_"+ str(C.early_stop)+"_"+str(C.regularizers)+"_.npy"

np.save(results_location+'/'+result_file, results_mean_iou)
np.save(results_location+'/'+test_result_file, test_results_mean_iou)
np.save(results_location+'/'+"num_samples", np.array(num_samples))

# plt.plot(num_samples,test_results_mean_iou, label ="Val Prediction")
# plt.plot(num_samples,results_mean_iou, label ='Training Prediction')
# plt.ylabel('Mean IOU')
# plt.xlabel('Number of Samples')
# plt.legend()
# plt.grid()
# plt.show()
