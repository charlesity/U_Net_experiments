
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
parser.add_argument("-ep", "--epochs",  required = True, help ="Number of epochs")
parser.add_argument("-dr_prob", "--dropout_prob",  required = True, help ="dropout_prob")
parser.add_argument("-exp_index", "--experiment_index_number",  required = True, help ="Index number of experiment")
parser.add_argument("-n_classes", "--num_classes",  required = True, help ="Number of classes")

parser = parser.parse_args()
log_filename = parser.log_filename

C = Config()

C.set_enable_dropout(parser.dropout_type)
C.epochs = int(parser.epochs)
C.standard_dropout = float(parser.dropout_prob)
C.num_classes = int(parser.num_classes)
dataset_location = parser.dataset_location

#prepare data augmentation
data_gen_args = dict(rotation_range= 90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip = True,
                     vertical_flip = True,
                     zoom_range=0.2)

all_images = glob.glob(dataset_location+'/image/*.png')
all_masks = glob.glob(dataset_location+'/label/*.png')


# training_ratios = np.arange(.99, .01, -.10)
training_ratios = [.20]
training_loss = np.zeros((len(training_ratios), C.epochs))

val_loss = np.zeros((len(training_ratios), C.epochs))
val_loss_mcmc = np.zeros((len(training_ratios)))

training_dice_coef = np.zeros((len(training_ratios), C.epochs))
val_dice_coef = np.zeros((len(training_ratios), C.epochs))
val_dice_coef_mcmc = np.zeros((len(training_ratios)))

standard_loss = np.zeros((len(training_ratios), C.epochs))
standard_dice = np.zeros((len(training_ratios), C.epochs))

dataset_sizes_used = []

i = int(parser.experiment_index_number)

print("Experiment number ", i)
C.randomSeed= np.random.randint(10) # start with a random seed for each experiment so we get a random shuffle each time
Train_images, Val_images, Train_masks, Val_masks = train_test_split(all_images, all_masks, test_size=0.33, random_state=C.randomSeed)
Val_dataframe = pd.DataFrame(list(zip(Val_images, Val_masks)))


#create the validation generator
val_generator = get_generator(data_gen_args, Val_dataframe, C)

for j, ratio in enumerate(training_ratios):

    try:
        #pick a subset of the training examples and train the model
        sub_train_images, _, sub_train_masks, _ = train_test_split(Train_images, Train_masks, test_size=ratio, random_state=C.randomSeed)
        print ("Sub training number ", j+1, " of ", len(training_ratios), "sample sizes, in experiment number ", i)
        dataset_sizes_used.append(len(sub_train_images))
    except:
        continue #if empty train exception(most likely) then continue
    sub_train_dataframe = pd.DataFrame(list(zip(sub_train_images, sub_train_masks)))
    sub_train_generator = get_generator(data_gen_args, sub_train_dataframe, C)




    network = model_instance(C)
    model = network.getModel()
    # model.summary()

    history = model.fit_generator(generator = sub_train_generator, steps_per_epoch=(len(sub_train_images)//C.batch_size) + 1, epochs=C.epochs, use_multiprocessing = True
                                  ,validation_data = val_generator, validation_steps = (len(Val_images)//C.batch_size)+1)
    training_loss[j] = history.history['loss']
    val_loss[j] = history.history['val_loss']
    training_dice_coef[j] = history.history['dice_coef']
    val_dice_coef[j] = history.history['val_dice_coef']


    mcmc_loss, mcmc_dice = network.stochastic_evaluate_generator(val_generator, C, (len(Val_images)//C.batch_size)+1)
    st_loss, st_dice = model.evaluate_generator(val_generator, steps=(len(Val_images)//C.batch_size)+1, use_multiprocessing=True)
    val_loss_mcmc[j] = mcmc_loss
    val_dice_coef_mcmc[j] = mcmc_dice
    standard_loss[j] = st_loss
    standard_dice[j] = st_dice

    # test images

    test_img, test_mask = next(val_generator)

    pred = model.predict(test_img)
    pred_st = network.stochastic_predict(test_img, C)


    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex= True, sharey = True)
    ax1.imshow(test_img[0,:,:,:])
    ax2.imshow(get_colored_segmentation_image(pred_st[0, :, :, :], 2))
    ax3.imshow(get_colored_segmentation_image(test_mask[0, :, :, :], 2))



    ax4.imshow(test_img[0,:,:,:])
    ax5.imshow(get_colored_segmentation_image(pred[0, :, :, :], 2))
    ax6.imshow(get_colored_segmentation_image(test_mask[0, :, :, :], 2))

    plt.show()
    print (mcmc_loss, mcmc_dice, st_loss, st_dice)


    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(history.history['loss'], label = 'training loss')
    ax1.plot(history.history['val_loss'], label ='val_loss')
    ax1.set_yscale('log')
    ax1.legend()

    ax2.plot(history.history['dice_coef'], label = 'training_dice_coef')
    ax2.plot(history.history['val_dice_coef'], label = 'val_dice_coef')
    ax2.set_yscale('log')
    ax2.legend()

    plt.show()
    exit()



theArrayLog = np.array({"training_loss": training_loss,
        "val_loss": val_loss, "val_loss_mcmc":val_loss_mcmc, "training_dice_coef":
            training_dice_coef, "val_dice_coef":val_dice_coef, "val_dice_coef_mcmc":val_dice_coef_mcmc,
            "dataset_sizes_used": dataset_sizes_used,"standard_loss":standard_loss, "standard_dice":standard_dice, "num_classes": C.num_classes})

try:
    savedArrayLog = np.load("./results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(C.epochs)+".npy", allow_pickle = True)
    #append here
    theArrayLog = np.append(savedArrayLog, theArrayLog)
    np.save("results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(C.epochs), theArrayLog)

except IOError as e:
    np.save("results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(C.epochs), theArrayLog)
