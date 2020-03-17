
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
parser.add_argument("-dt", "--dropout_type", type=int, required=True,   help="NoDropout=0, Standard dropout= 1, max_pool=2, Both=3")
parser.add_argument("-lf", "--log_filename",  required = True, help ="file name for the logs")

parser = parser.parse_args()
log_filename = parser.log_filename

C = Config()

C.set_enable_dropout(parser.dropout_type)
C.num_classes = 2
dataset_location = parser.dataset_location

#prepare data augmentation
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range= 90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip = True,
                     vertical_flip = True,
                     zoom_range=0.2)
all_images = glob.glob(dataset_location+'/train/image/*.png')
all_masks = glob.glob(dataset_location+'/train/label/*.png')


training_ratios = np.arange(.99, .01, -.10)
# training_ratios = [.33]
training_loss = np.zeros((C.n_experiments, len(training_ratios), C.epochs))

val_loss = np.zeros((C.n_experiments,len(training_ratios), C.epochs))
val_loss_mcmc = np.zeros((C.n_experiments,len(training_ratios)))

training_dice_coef = np.zeros((C.n_experiments,len(training_ratios), C.epochs))
val_dice_coef = np.zeros((C.n_experiments,len(training_ratios), C.epochs))
val_dice_coef_mcmc = np.zeros((C.n_experiments,len(training_ratios)))

dataset_sizes_used = []

for i in range(C.n_experiments):
        print("Experiment number ", i+1)
        C.randomSeed= np.random.randint(10) # start with a random seed for each experiment so we get a random shuffle each time
        Train_images, Val_images, Train_masks, Val_masks = train_test_split(all_images, all_masks, test_size=0.33, random_state=C.randomSeed)
        Val_dataframe = pd.DataFrame(list(zip(Val_images, Val_masks)))

        #prepare the sample images for the generators
        img1 = cv2.imread(Val_images[0])
        sample_images = np.zeros(shape=(2, *img1.shape))
        sample_images[0] = img1
        sample_images[1] = cv2.imread(Val_images[0])

        msk1 = cv2.imread(Val_masks[0])
        sample_mask = np.zeros(shape=(2, *msk1.shape))

        sample_mask[0] = msk1
        sample_mask[1] = cv2.imread(Val_masks[1])

        #create the validation generator
        val_generator = get_generator(data_gen_args, Val_dataframe, C, (sample_images, sample_mask))

        for j, ratio in enumerate(training_ratios):

            try:
                #pick a subset of the training examples and train the model
                sub_train_images, _, sub_train_masks, _ = train_test_split(Train_images, Train_masks, test_size=ratio, random_state=C.randomSeed)
                print ("Sub training number ", j+1, " of ", len(training_ratios), "sample sizes, in experiment number ", i+1)
                dataset_sizes_used.append(len(sub_train_images))
            except:
                continue #if empty train exception(most likely) then continue
            sub_train_dataframe = pd.DataFrame(list(zip(sub_train_images, sub_train_masks)))
            sub_train_generator = get_generator(data_gen_args, sub_train_dataframe, C, (sample_images, sample_mask))


            network = model_instance(C)
            model = network.getModel()

            history = model.fit_generator(generator = sub_train_generator, steps_per_epoch=(len(sub_train_images)//C.batch_size) + 1, epochs=C.epochs, use_multiprocessing = True
                                          ,validation_data = val_generator, validation_steps = 2*((len(Val_images)//C.batch_size)+1))
            training_loss[i, j] = history.history['loss']
            val_loss[i, j] = history.history['val_loss']
            training_dice_coef[i, j] = history.history['dice_coef']
            val_dice_coef[i, j] = history.history['val_dice_coef']


            mcmc_loss, mcmc_dice = network.stochastic_evaluate_generator(val_generator, C, 2*((len(Val_images)//C.batch_size)+1))
            val_loss_mcmc[i, j] = mcmc_loss
            val_dice_coef_mcmc = mcmc_dice

            
            # test images
            # test_img, test_mask = next(val_generator)
            # pred = model.predict(test_img)
            # pred_st = network.stochastic_predict(test_img, C)
            #
            # f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex= True, sharey = True)
            # ax1.imshow(test_img[0,:,:,:])
            # ax2.imshow(get_colored_segmentation_image(pred_st[0, :, :, :], 2))
            # ax3.imshow(get_colored_segmentation_image(test_mask[0, :, :, :], 2))
            #
            #
            # ax4.imshow(test_img[0,:,:,:])
            # ax5.imshow(get_colored_segmentation_image(pred[0, :, :, :], 2))
            # ax6.imshow(get_colored_segmentation_image(test_mask[0, :, :, :], 2))
            #
            # plt.show()
            # exit()
            # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            # ax1.plot(history.history['loss'], label = 'training loss')
            # ax1.plot(history.history['val_loss'], label ='val_loss')
            # ax1.legend()
            #
            # ax2.plot(history.history['dice_coef'], label = 'training_dice_coef')
            # ax2.plot(history.history['val_dice_coef'], label = 'val_dice_coef')
            # ax2.legend()





logs = {"training_loss": training_loss,
        "val_loss": val_loss, "val_loss_mcmc":val_loss_mcmc, "training_dice_coef":
            training_dice_coef, "val_dice_coef":val_dice_coef, "val_dice_coef_mcmc":val_dice_coef_mcmc,
            "dataset_sizes_used": dataset_sizes_used,
            "num_experiments": C.n_experiments, "num_classes": C.num_classes}
np.save("results/"+log_filename+str(parser.dropout_type), logs)
