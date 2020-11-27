#Charles I. Saidu
import random
import warnings
import time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import glob as glob

from utilities import *

from Config import Config

import argparse

start_time = time.time()
parser = argparse.ArgumentParser()

parser.add_argument("-ds", "--dataset_location",  required = True, help ="link to the datasource")
parser.add_argument("-weights", "--stored_weights",  required = True, help ="link to the stored weights")
parser.add_argument("-f_ext", "--file_ext",  required = True, help ="File Extension")
parser.add_argument("-n_classes", "--num_classes",  required = True, help ="Number of classes")
parser = parser.parse_args()

C = Config()


C.set_enable_dropout(2) #Max pool dropout plus batnorm
C.standard_dropout = .5 #dropout probability
file_extension = parser.file_ext
if (int(parser.num_classes) <= 2):
    C.num_classes = 2
else:
    C.num_classes = int(parser.num_classes)



dataset_location = parser.dataset_location

all_images = glob.glob(dataset_location+'/dataset/image/*.'+file_extension)
all_masks = glob.glob(dataset_location+'/dataset/label/*.'+file_extension)

C.randomSeed= np.random.randint(10) # start with a random seed for each experiment so we get a random shuffle each time
Train_images, Val_images, Train_masks, Val_masks = train_test_split(all_images, all_masks, test_size=0.33, random_state=C.randomSeed)

Val_dataframe = pd.DataFrame(list(zip(Val_images, Val_masks)))
val_generator = get_generator(Val_dataframe, C)

network = model_instance(C)
model = network.getModel()

model.load_weights(parser.stored_weights)
# test images
test_img, test_mask = next(val_generator)
pred = model.predict(test_img)
pred_st = network.stochastic_predict(test_img, C)

f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex= True, sharey = True)
ax1.imshow(test_img[0,:,:,:])
ax1.set_title("Image")
ax2.imshow(get_colored_segmentation_image(pred_st[0, :, :, :], C.num_classes))
ax2.set_title("MCMC Prediction")
ax3.imshow(get_colored_segmentation_image(test_mask[0, :, :, :], C.num_classes))
ax3.set_title("True Label")

get_uncertain_segmentation_image(test_mask[0, :, :, :], C.num_classes)


plt.show()
