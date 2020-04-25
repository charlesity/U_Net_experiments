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
parser.add_argument("-dt", "--dropout_type", type=int, required=True,   help="BatNorm=0, Standard dropout= 1, max_pool=2, Both=3")
parser.add_argument("-lf", "--log_filename",  required = True, help ="file name for the logs")
parser.add_argument("-ep", "--epochs",  required = True, help ="Number of epochs")
parser.add_argument("-dr_prob", "--dropout_prob",  required = True, help ="dropout_prob")
parser.add_argument("-exp_index", "--experiment_index_number",  required = True, help ="Index number of experiment")
parser.add_argument("-n_classes", "--num_classes",  required = True, help ="Number of classes")
parser.add_argument("-f_ext", "--file_ext",  required = True, help ="File Extension")
parser.add_argument("-ac_batch", "--active_batch",  required = True, help ="File Extension")
parser.add_argument("-qt", "--q_type", type=int,  help="1 = Entropy, 2 = Committee-KL, 3 =Committee-Jen,  else = Random acquition")
parser.add_argument("-re_w", "--re_initialize_weights", type=int, help="determine whether or not weights should be re-initialized")
parser.add_argument("-early_s", "--early_stop", type=int,  help="Indicates early stop")
parser.add_argument("-act_ac_ind", "--active_acquisition_index", type=int,  help="Index of Active acquisition")

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

C.early_stop = int(parser.early_stop)
acquisition_type = int(parser.q_type)
dataset_location = parser.dataset_location
weight_reinit = parser.re_initialize_weights

active_acquisition_index = parser.active_acquisition_index

exp_index_num = int(parser.experiment_index_number)

dataset_sizes_used = []

training_loss = []
val_loss = []
val_loss_mcmc = []

training_dice_coef = []
val_dice_coef = []
val_dice_coef_mcmc = []

first_Training = False
print("Experiment number ", exp_index_num)


#log selected image files
try:
    Val_dataframe = pd.read_pickle("./active_dataframe_log/"+log_filename+str(parser.dropout_type)+str(acquisition_type)+str(exp_index_num)+"_val_dataframe_"+".pkl")
    active_train_dataframe = pd.read_pickle("./active_dataframe_log/"+log_filename+str(parser.dropout_type)+str(acquisition_type)+str(exp_index_num)+str(active_acquisition_index)+"_active_train_dataframe_"+".pkl")
    unlabeled_dataframe= pd.read_pickle("./active_dataframe_log/"+log_filename+str(parser.dropout_type)+str(acquisition_type)+str(exp_index_num)+str(active_acquisition_index)+"_unlabeled_dataframe_"+".pkl")
    print ("DataFrames Available")
except Exception as e:
    print ("DataFrames NOT Available")
    print(e)
    if active_acquisition_index > 0:
        print("No more unlabeled Dataset")
        exit()
    all_images = glob.glob(dataset_location+'/dataset/image/*.'+file_extension)
    all_masks = glob.glob(dataset_location+'/dataset/label/*.'+file_extension)

    C.randomSeed= np.random.randint(10) # start with a random seed for each experiment so we get a random shuffle each time
    Train_images, Val_images, Train_masks, Val_masks = train_test_split(all_images, all_masks, test_size=0.33, random_state=C.randomSeed)
    active_train_images = Train_images[:C.active_batch]
    active_train_masks = Train_masks[:C.active_batch]

    #micmic unlabeled set with already labeled data
    unlabeled_images = Train_images[C.active_batch:]
    unlabeled_masks = Train_masks[C.active_batch:]

    Val_dataframe = pd.DataFrame(list(zip(Val_images, Val_masks)))
    active_train_dataframe = pd.DataFrame(list(zip(active_train_images, active_train_masks)))
    unlabeled_dataframe = pd.DataFrame(list(zip(unlabeled_images, unlabeled_masks)))

    Val_dataframe.to_pickle("./active_dataframe_log/"+log_filename+str(parser.dropout_type)+str(acquisition_type)+str(exp_index_num)+"_val_dataframe_"+".pkl")
    active_train_dataframe.to_pickle("./active_dataframe_log/"+log_filename+str(parser.dropout_type)+str(acquisition_type)+str(exp_index_num)+str(active_acquisition_index)+"_active_train_dataframe_"+".pkl")
    unlabeled_dataframe.to_pickle("./active_dataframe_log/"+log_filename+str(parser.dropout_type)+str(acquisition_type)+str(exp_index_num)+str(active_acquisition_index)+"_unlabeled_dataframe_"+".pkl")

    first_Training = True

# print (Val_dataframe.head())
# print (active_train_dataframe.head())
# print (unlabeled_dataframe.head())
#create the validation generator
val_generator = get_generator(Val_dataframe, C)


network = model_instance(C)
model = network.getModel()
es = EarlyStopping(monitor='val_loss', mode='auto', patience = C.early_stop_patience, verbose=1) if C.early_stop == 1 else None

print ("Active acquition ", active_acquisition_index, " in experiment num ", exp_index_num, " of Dropout type ", parser.dropout_type)

if first_Training:
    print ("First Training")
    unlabeled_generator = get_generator_with_filename(unlabeled_dataframe, C)
    active_train_generator = get_generator(active_train_dataframe, C)
    dataset_sizes_used.append(len(active_train_dataframe))

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

    #     plt.show()
    # exit()
    history = model.fit_generator(generator = active_train_generator, steps_per_epoch=(len(active_train_dataframe)//C.batch_size) + 1, epochs=C.epochs, use_multiprocessing = True
                                  ,validation_data = val_generator, validation_steps = (len(Val_dataframe)//C.batch_size)+1, callbacks=[es])
    #save_the_model
    model.save_weights("./stored_weights/"+log_filename+"_weights_"+str(parser.dropout_type)+str(acquisition_type)+".h5")

    unlabeled_scores_dict = score_unlabeled_images(acquisition_type,(len(unlabeled_dataframe)//C.batch_size) + 1, unlabeled_generator, network, C)

else:
    try:

        unlabeled_generator = get_generator_with_filename(unlabeled_dataframe, C)
        active_train_generator = get_generator(active_train_dataframe, C)
        dataset_sizes_used.append(len(active_train_dataframe))

        if(len(unlabeled_dataframe)<=0):
            print("Exiting Program because there are no unlabeled images. Number is ", len(unlabeled_dataframe))
            exit()

        # print (dataset_sizes_used) TODO
        if weight_reinit == 1:
            history = model.fit_generator(generator = active_train_generator, steps_per_epoch=(len(active_train_dataframe)//C.batch_size) + 1, epochs=C.epochs, use_multiprocessing = True
                                      ,validation_data = val_generator, validation_steps = (len(Val_dataframe)//C.batch_size)+1, callbacks=[es])
        else:
            # #fine tune weights model by loading previous weights
            print("Loaded weight is ", "./stored_weights/"+log_filename+"_weights_"+str(exp_index_num)+".h5")
            model.load_weights("./stored_weights/"+log_filename+"_weights_"+str(parser.dropout_type)+str(acquisition_type)+".h5")
            history = model.fit_generator(generator = active_train_generator, steps_per_epoch=(len(active_train_dataframe)//C.batch_size) + 1, epochs=C.epochs, use_multiprocessing = True
                                      ,validation_data = val_generator, validation_steps = (len(Val_dataframe)//C.batch_size)+1, callbacks=[es])
        #save_the_model
        model.save_weights("./stored_weights/"+log_filename+"_weights_"+str(parser.dropout_type)+str(acquisition_type)+".h5")
        #score the unlabeled dataset

        unlabeled_scores_dict = score_unlabeled_images(acquisition_type,(len(unlabeled_dataframe)//C.batch_size) + 1, unlabeled_generator, network, C)

    except Exception as e:
        print (e)
        print("No further active train file to read. Program terminating")


print("Previous Size of Unlabeled and Active Training ", len(unlabeled_dataframe), len(active_train_dataframe))
# select the most informative
most_informative = unlabeled_dataframe.loc[unlabeled_dataframe[0].isin([i[0] for i in unlabeled_scores_dict[:C.active_batch]]) ]
#add to the active training set
active_train_dataframe = pd.concat([active_train_dataframe, most_informative])
#remove from the unlabeled set
unlabeled_dataframe = unlabeled_dataframe.loc[~unlabeled_dataframe[0].isin(most_informative[0])]
print("New Size of Unlabeled and Active Training ", len(unlabeled_dataframe), len(active_train_dataframe), " about to be saved")



#collect logs
training_loss.append(history.history['loss'][-1])
val_loss.append(history.history['val_loss'][-1])
training_dice_coef.append(history.history['dice_coef'][-1])
val_dice_coef.append(history.history['val_dice_coef'][-1])

print ("Evaluating mcmc metrics")
mcmc_loss, mcmc_dice = network.stochastic_evaluate_generator(val_generator, C, (len(Val_dataframe)//C.batch_size)+1)
print ("Done evaluating mcmc metrics")
# st_loss, st_dice = model.evaluate_generator(val_generator, steps=(len(Val_images)//C.batch_size)+1, use_multiprocessing=True)
val_loss_mcmc.append(mcmc_loss)
val_dice_coef_mcmc.append(mcmc_dice)

print("Done with acquisition no. ", active_acquisition_index, "out of ", len(unlabeled_dataframe)//C.active_batch, "remaining acquisitions in experiment number ", exp_index_num, " of Dropout type ", parser.dropout_type)

#save the unlabeled dataframe and the active train dataframe for next acquisition
active_acquisition_index +=1
active_train_dataframe.to_pickle("./active_dataframe_log/"+log_filename+str(parser.dropout_type)+str(acquisition_type)+str(exp_index_num)+str(active_acquisition_index)+"_active_train_dataframe_"+".pkl")
unlabeled_dataframe.to_pickle("./active_dataframe_log/"+log_filename+str(parser.dropout_type)+str(acquisition_type)+str(exp_index_num)+str(active_acquisition_index)+"_unlabeled_dataframe_"+".pkl")


#prepare record log file

try:
    savedArrayLog = np.load("./results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(acquisition_type)+"_"+str(weight_reinit)+"_"+str(C.early_stop)+"_"+str(exp_index_num)+".npy", allow_pickle = True)
    #pick up previous record files
    n_training_loss = savedArrayLog[0][1].tolist() + training_loss
    n_val_loss = savedArrayLog[0][2].tolist() + val_loss
    n_val_loss_mcmc = savedArrayLog[0][3].tolist() + val_loss_mcmc
    n_training_dice_coef = savedArrayLog[0][4].tolist() + training_dice_coef
    n_val_dice_coef = savedArrayLog[0][5].tolist() + val_dice_coef
    n_val_dice_coef_mcmc = savedArrayLog[0][6].tolist() + val_dice_coef_mcmc
    n_dataset_sizes_used = savedArrayLog[0][7].tolist() + dataset_sizes_used

    print ("Database sizes used", n_dataset_sizes_used)
    dt = [('exp_num', 'i4')]+ [('log'+str(l), 'f4', np.shape(n_training_loss)) for l in range(7)]

    #append new log
    logs_to_save = [(exp_index_num, n_training_loss, n_val_loss, n_val_loss_mcmc, n_training_dice_coef, n_val_dice_coef, n_val_dice_coef_mcmc, n_dataset_sizes_used)]
    newLogToSave = np.array(logs_to_save, dtype = dt)
    #save structured array
    np.save("results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(acquisition_type)+"_"+str(weight_reinit)+"_"+str(C.early_stop)+"_"+str(exp_index_num), newLogToSave)

except IOError as e:
    print ("Error File Not Exists: So we save the first one")
    #This exception is caught if log file doesnt exist
    dt = [('exp_num', 'i4')]+ [('log'+str(l), 'f4', np.shape(training_loss)) for l in range(7)]
    theArrayLog = np.array([(exp_index_num, training_loss, val_loss, val_loss_mcmc, training_dice_coef, val_dice_coef, val_dice_coef_mcmc, dataset_sizes_used)],
                           dtype=dt)
    print ("Database sizes used", dataset_sizes_used)
    np.save("results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(acquisition_type)+"_"+str(weight_reinit)+"_"+str(C.early_stop)+"_"+str(exp_index_num), theArrayLog)
#
#
#










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
