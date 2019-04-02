
import random
import warnings
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import argparse
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
from sklearn.utils import shuffle

from utilities import *

from Config import Config

import argparse


start_time = time.time()
parser = argparse.ArgumentParser()

parser.add_argument("-ds", "--dataset_location",  required = True, help ="link to the datasource")
parser.add_argument("-dt", "--dropout_type", type=int, required=True,   help="Standard=0, max_pool=1, both=2")

parser = parser.parse_args()

C = Config()
C.set_enable_dropout(parser.dropout_type)

random.seed = C.seed
np.random.seed = C.seed



dataset_location = parser.dataset_location

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

val_dataset_sizes = np.arange(.99, .01, -.10)

for i in range(C.n_experiments):
    train_ids = shuffle(train_ids)
    loss = []
    val_loss = []
    mean_iou = []
    val_mean_iou = []
    loss_np = np.zeros((len(val_dataset_sizes),  C.epoch))
    val_loss_np = np.zeros((len(val_dataset_sizes),  C.epoch))
    mean_iou_np = np.zeros((len(val_dataset_sizes),  C.epoch))
    val_mean_iou_np = np.zeros((len(val_dataset_sizes),  C.epoch))
    for val_ratio in val_dataset_sizes:
        C.validation_split = val_ratio
        model = model_instance(C)

        # history = model.fit(X_train, Y_train, validation_split=C.validation_split, batch_size=C.batch_size, epochs=C.epoch)
        history = model.fit(X_train, Y_train, validation_split=C.validation_split,  epochs=C.epoch)
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        mean_iou.append(history.history['mean_iou'])
        val_mean_iou.append(history.history['val_mean_iou'])

    loss_np += np.array(loss)
    val_loss_np += np.array(val_loss)
    mean_iou_np += np.array(mean_iou)
    val_mean_iou_np +=np.array(val_mean_iou)


loss_np /= C.n_experiments
val_loss_np /= C.n_experiments
mean_iou_np /= C.n_experiments
val_mean_iou_np /= C.n_experiments
R= np.stack((loss_np, val_loss_np, mean_iou_np, mean_iou_np))
file = switch_result_file(parser.dropout_type)
np.save('results/'+file, R)
