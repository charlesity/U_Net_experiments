#Code written by Charles I. Saidu https://github.com/charlesity
#All rights reserved
import math
import numpy as np
import argparse
import sys


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
from sklearn.metrics import jaccard_similarity_score # for mean_iou

from keras.utils import plot_model

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


arg = parser.parse_args()
C = Config()
C.set_enable_dropout(2)  # max_pool dropout type
C.regularizers = bool(arg.regularizers)

random.seed = C.seed
np.random.seed = C.seed
dataset_location = arg.dataset_location
C.early_stop = arg.early_stop

# directory to save result
script_file = os.path.splitext(os.path.basename(__file__))[0]

if not os.path.exists('./results/'+script_file):
    results_location = './results/'+script_file
    os.mkdir(results_location)
else:
    results_location = './results/'+script_file

test_results_mean_iou = np.zeros((C.n_experiments, C.num_active_queries+1))
results_mean_iou = np.zeros((C.n_experiments, C.num_active_queries+1))  #in-sample mean_iou

num_samples = []
if (os.path.isfile(dataset_location+'/X_train_bowl.npy')):
    X_train = np.load(dataset_location+'/X_train_bowl.npy')
    Y_train = np.load(dataset_location+'/Y_train_bowl.npy')
    X_test = np.load(dataset_location+'/X_test_bowl.npy')

    train_ids =np.arange(0, X_train.shape[0])
    test_ids = np.arange(0, X_test.shape[0])

else:
    train_ids = list(next(os.walk(C.TRAIN_PATH))[1])
    test_ids = list(next(os.walk(C.TEST_PATH))[1])

    X_train, Y_train, X_test = getAndResizeMask(train_ids, test_ids, C)
    np.save(dataset_location+'/X_train_bowl', X_train)
    np.save(dataset_location+'/Y_train_bowl', Y_train)
    np.save(dataset_location+'/X_test_bowl', X_test)

#set the shape configuration
C.IMG_HEIGHT, C.IMG_WIDTH, C.IMG_CHANNELS = X_train[0].shape[0], X_train[0].shape[1], X_train[0].shape[2]

#set asside test set with mask for benchmarking results


for e in range(C.n_experiments):
    all_indices  = np.arange(X_train.shape[0])
    np.random.shuffle(all_indices)

    ms = int(C.mask_test_set_ratio * X_train.shape[0])
    mask_test_set_indices = all_indices[:ms]

    X_test_with_mask = X_train[mask_test_set_indices]
    y_test_mask = Y_train[mask_test_set_indices]



    ac_ind = int(C.initial_training_ratio * (X_train.shape[0] - ms))   #  (X_train.shape[0] - ms) becase new training set is X_train - number reserved as test set.
    active_indices = all_indices[ms: (ms +ac_ind)].copy()


    un_labeled_index = all_indices[ms + ac_ind:].copy()

    active_train_X = X_train[active_indices]
    active_train_y = Y_train[active_indices]

    unlabeled_X = X_train[un_labeled_index]
    unlabeled_y = Y_train[un_labeled_index]

    # print (X_test_with_mask.shape, y_test_mask.shape)
    # print (active_train_X.shape, active_train_y.shape)
    # print (unlabeled_X.shape)
    # exit()
    C.output_shape = (active_train_y.shape[1:])

    net_instance = model_instance(C)

    # model.summary()
    # plot_model(net_instance.getModel(), to_file='fcn_dropout')
    # model_to_dot(net_instance.getModel()).create(prog='dot', format ='svg')
    # exit()


    if C.early_stop == 1:
        es = EarlyStopping(monitor='val_mean_iou', mode='auto',patience=C.early_stop_patience, verbose=1)
        history = net_instance.getModel().fit(active_train_X, active_train_y, validation_split=C.validation_split
                                              ,epochs=C.epoch, callbacks=[es])
    else:
        history = net_instance.getModel().fit(active_train_X, active_train_y, validation_split=C.validation_split
                                              ,epochs=C.epoch)

    if arg.mc_prediction == 0:
        pred_mask = net_instance.getModel().predict(active_train_X)  #training sample prediction
        test_pred_mask = net_instance.getModel().predict(X_test_with_mask)
    else:
        pred_mask = net_instance.stochastic_predict(active_train_X, C)
        test_pred_mask = net_instance.stochastic_predict(X_test_with_mask, C)

    # #uncomment to visualize results iteratively
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
    #     ax.imshow(y_test_mask[i, :, :, 0])
    #     ax.set_title('Mask Test')
    #     idx+=1
    #     ax = fig.add_subplot(num_rows, 3, idx, xticks=[], yticks=[])
    #     p = test_pred_mask[i, :, :, 0]
    #     p[p >= C.mask_threshold] = 1
    #     p[p < C.mask_threshold] = 0  # threshold predicted mask
    #     ax.imshow(p)
    #     ax.set_title('Predicted Mask Test')
    # plt.show()

    #for training samples
    p_train_y = pred_mask.ravel()
    p_train_y[p_train_y >= C.mask_threshold] = 1
    p_train_y[p_train_y < C.mask_threshold] = 0

    # test predictions
    p_test = test_pred_mask.ravel()
    p_test[p_test >= C.mask_threshold] = 1
    p_test[p_test < C.mask_threshold] = 0




    train_mean_iou = jaccard_similarity_score(active_train_y.ravel().astype(int), p_train_y.ravel().astype(int))
    test_mean_iou = jaccard_similarity_score(y_test_mask.ravel().astype(int), p_test.ravel().astype(int))

    results_mean_iou[e][0] = train_mean_iou
    test_results_mean_iou[e][0] = test_mean_iou

    if e == 0:
        num_samples.append(active_train_X.shape[0])


    #retrieve weights in case weight re-initialization is set to 1
    current_weights = net_instance.getModel().get_weights()

    for aquisitions in range(C.num_active_queries):
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
            index_maximum = independent_pixel_entropy(net_instance, unlabeled_X, C)
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

        #uncomment to visualized 9 informative unlabeled and informative samples
        # fig = plt.figure(figsize=(4, 25))
        # idx = 0
        # num_rows = 5
        # for i in range(num_rows):
        #     idx +=1
        #     ax = fig.add_subplot(num_rows, 2, idx, xticks=[], yticks=[])
        #     ax.imshow(additional_samples_X[i, :, :, :])
        #     idx +=1
        #     ax = fig.add_subplot(num_rows, 2, idx, xticks=[], yticks=[])
        #     mask = net_instance.getModel().predict(np.expand_dims(additional_samples_X[i, :, :, :], axis=0))
        #     print (mask.shape)
        #     # exit()
        #     ax.imshow(mask[0, :, :, 0])
        # fig.suptitle("Most informative samples")
        # plt.show()
        #
        if arg.re_initialize_weights == 0:
            net_instance.getModel().set_weights(current_weights)  #weight fine-tuning
        else:
            net_instance = model_instance(C)  #re-initialize weights


        if C.early_stop == 1:
            es = EarlyStopping(monitor='val_mean_iou', mode='auto', patience = C.early_stop_patience, verbose=1)
            history = net_instance.getModel().fit(active_train_X, active_train_y, validation_split=C.validation_split
                                              ,epochs=C.epoch, callbacks=[es])
        else:
            history = net_instance.getModel().fit(active_train_X, active_train_y, validation_split=C.validation_split
                                                  ,epochs=C.epoch)

        if arg.mc_prediction == 0:
            pred_mask = net_instance.getModel().predict(active_train_X)  #training sample prediction
            test_pred_mask = net_instance.getModel().predict(X_test_with_mask)
        else:
            pred_mask = net_instance.stochastic_predict(active_train_X, C)
            test_pred_mask = net_instance.stochastic_predict(X_test_with_mask, C)

        exit()
        #uncomment for iterative visualization
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
        #     ax.imshow(y_test_mask[i, :, :, 0])
        #     ax.set_title('Mask Test')
        #     idx+=1
        #     ax = fig.add_subplot(num_rows, 3, idx, xticks=[], yticks=[])
        #     p = test_pred_mask[i, :, :, 0]
        #     p[p >= C.mask_threshold] = 1
        #     p[p < C.mask_threshold] = 0  # threshold predicted mask
        #     ax.imshow(p)
        #     ax.set_title('Predicted Mask Test')
        # plt.show()
        # test_mean_iou = jaccard_similarity_score(y_test_mask.ravel(), test_pred_mask.ravel().astype(int))
        # print (round(test_mean_iou, 3))

        #for training samples
        p_train_y = pred_mask.ravel()
        p_train_y[p_train_y >= C.mask_threshold] = 1
        p_train_y[p_train_y < C.mask_threshold] = 0

        # test predictions
        p_test = test_pred_mask.ravel()
        p_test[p_test >= C.mask_threshold] = 1
        p_test[p_test < C.mask_threshold] = 0


        train_mean_iou = jaccard_similarity_score(active_train_y.ravel().astype(int), p_train_y.ravel().astype(int))
        test_mean_iou = jaccard_similarity_score(y_test_mask.ravel().astype(int), p_test.ravel().astype(int))

        results_mean_iou[e][aquisitions+1] = train_mean_iou
        test_results_mean_iou[e][aquisitions+1] = test_mean_iou

        if e == 0:
            num_samples.append(active_train_X.shape[0])

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
