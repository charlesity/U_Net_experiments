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
parser.add_argument("-f_ext", "--file_ext",  required = True, help ="File Extension")
parser.add_argument("-ac_batch", "--active_batch",  required = True, help ="File Extension")

parser = parser.parse_args()
log_filename = parser.log_filename

C = Config()

C.active_batch = int(parser.active_batch)

C.set_enable_dropout(parser.dropout_type)
C.epochs = int(parser.epochs)
file_extension = parser.file_ext
C.standard_dropout = float(parser.dropout_prob)
if (int(parser.num_classes) <= 2):
    C.num_classes = 2
else:
    C.num_classes = int(parser.num_classes)

dataset_location = parser.dataset_location


all_images = glob.glob(dataset_location+'/dataset/image/*.'+file_extension)
all_masks = glob.glob(dataset_location+'/dataset/label/*.'+file_extension)


val_loss = None
val_loss_mcmc = None

training_dice_coef = None
val_dice_coef = None
val_dice_coef_mcmc = None

standard_loss = None
standard_dice = None

dataset_sizes_used = None

i = int(parser.experiment_index_number)


print("Experiment number ", i)
C.randomSeed= np.random.randint(10) # start with a random seed for each experiment so we get a random shuffle each time
Train_images, Val_images, Train_masks, Val_masks = train_test_split(all_images, all_masks, test_size=0.33, random_state=C.randomSeed)
Val_dataframe = pd.DataFrame(list(zip(Val_images, Val_masks)))

#create the validation generator
val_generator = get_generator(Val_dataframe, C)

active_train_images = Train_images[:C.active_batch]
active_train_masks = Train_masks[:C.active_batch]

#micmic unlabeled set with already labeled data
unlabeled_images = Train_images[:C.active_batch]
unlabeled_masks = Train_masks[:C.active_batch]

first_result = True # needed to stackup logs after first active learning results
while len(unlabeled_images) > 0:
    unlabeled_scores = list()
    active_train_dataframe = pd.DataFrame(list(zip(active_train_images, active_train_masks)))
    active_train_generator = get_generator(active_train_dataframe, C)

    unlabeled_dataframe = pd.DataFrame(list(zip(unlabeled_images, unlabeled_masks)))
    unlabeled_generator = get_generator_with_filename(unlabeled_dataframe, C)

    #uncomment to visualization images from generator with filenames
    # for i in range ((len(active_train_images)//C.batch_size) +1):
    #     img, mask, fn_imgs, fn_masks = next(unlabeled_generator)
    #     fig, ax = plt.subplots(img.shape[0],2)
    #     for j in range(img.shape[0]):
    #         m_coloured = get_colored_segmentation_image(mask[j], C.num_classes)
    #         print ("maximum image pixel "+str(img[j].max()), "maximum mask pixel "+str(m_coloured.max()))
    #         ax[j][0].imshow(img[j])
    #         ax[j][0].title.set_text(fn_imgs[j][-12:])
    #         ax[j][1].imshow(m_coloured.astype(np.uint8))
    #         ax[j][1].title.set_text(fn_masks[j][-12:])
    #
    #     plt.show()
    # exit()


    network = model_instance(C)
    model = network.getModel()
    # # model.summary()
    #
    history = model.fit_generator(generator = active_train_generator, steps_per_epoch=(len(active_train_dataframe)//C.batch_size) + 1, epochs=C.epochs, use_multiprocessing = True
                                  ,validation_data = val_generator, validation_steps = (len(Val_images)//C.batch_size)+1)

    for it in range((len(active_train_dataframe)//C.batch_size) + 1):
        img, _, fn_imgs, _ = next(unlabeled_generator)
        out_shape = model.layers[-1].output.get_shape().as_list()[1:]
        stochastic_predictions = np.zeros((C.dropout_iterations,img.shape[0], *out_shape))
        for i in range(C.dropout_iterations):
            stochastic_predictions[i] = network.stochastic_foward_pass(img)
        print (stochastic_predictions.shape, stochastic_predictions[0, 0,:,:,:].shape)
        exit()







    # calculate acquitive query score for unlabeled set

    # if first_result:
    #     training_loss =history.history['loss']
    #     val_loss = history['val_loss']]
    #     first_result = False
    # else:
    #     training_loss =np.stack(training_loss, history.history['loss'])

    # training_dice_coef[j] = history.history['dice_coef']
    # val_dice_coef[j] = history.history['val_dice_coef']
    #
    #
    # mcmc_loss, mcmc_dice = network.stochastic_evaluate_generator(val_generator, C, (len(Val_images)//C.batch_size)+1)
    # st_loss, st_dice = model.evaluate_generator(val_generator, steps=(len(Val_images)//C.batch_size)+1, use_multiprocessing=True)
    # val_loss_mcmc[j] = mcmc_loss
    # val_dice_coef_mcmc[j] = mcmc_dice
    # standard_loss[j] = st_loss
    # standard_dice[j] = st_dice


    # # test images
    # test_img, test_mask = next(val_generator)
    # pred = model.predict(test_img)
    # pred_st = network.stochastic_predict(test_img, C)
    #
    # f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex= True, sharey = True)
    # ax1.imshow(test_img[0,:,:,:])
    # ax2.imshow(get_colored_segmentation_image(pred_st[0, :, :, :], C.num_classes))
    # ax3.imshow(get_colored_segmentation_image(test_mask[0, :, :, :], C.num_classes))
    #
    #
    #
    # ax4.imshow(test_img[0,:,:,:])
    # ax5.imshow(get_colored_segmentation_image(pred[0, :, :, :], C.num_classes))
    # ax6.imshow(get_colored_segmentation_image(test_mask[0, :, :, :], C.num_classes))
    #
    # plt.show()
    # print (mcmc_loss, mcmc_dice, st_loss, st_dice)
    #
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(history.history['loss'], label = 'training loss')
    # ax1.plot(history.history['val_loss'], label ='val_loss')
    # ax1.set_yscale('log')
    # ax1.legend()
    #
    # ax2.plot(history.history['dice_coef'], label = 'training_dice_coef')
    # ax2.plot(history.history['val_dice_coef'], label = 'val_dice_coef')
    # ax2.set_yscale('log')
    # ax2.legend()
    #
    # plt.show()
    # exit()
