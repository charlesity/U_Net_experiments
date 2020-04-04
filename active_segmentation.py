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
import cynthonized_functions



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
parser.add_argument("-qt", "--q_type", type=int,  help="1 = Entropy, 2 = Committee-KL, 3 =Committee-Jen,  else = Random acquition")
parser.add_argument("-early_s", "--early_stop", type=int,  help="Indicates early stop")


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



training_loss = dict()
val_loss = dict()
val_loss_mcmc = []

training_dice_coef = dict()
val_dice_coef = dict()
val_dice_coef_mcmc = []


print("Experiment number ", i)
C.randomSeed= np.random.randint(10) # start with a random seed for each experiment so we get a random shuffle each time
Train_images, Val_images, Train_masks, Val_masks = train_test_split(all_images, all_masks, test_size=0.33, random_state=C.randomSeed)
Val_dataframe = pd.DataFrame(list(zip(Val_images, Val_masks)))

#create the validation generator
val_generator = get_generator(Val_dataframe, C)

active_train_images = Train_images[:C.active_batch]
active_train_masks = Train_masks[:C.active_batch]

#micmic unlabeled set with already labeled data
unlabeled_images = Train_images[C.active_batch:]
unlabeled_masks = Train_masks[C.active_batch:]

first_result = True # needed to stackup logs after first active learning results
acquisitions = 0
active_train_dataframe = pd.DataFrame(list(zip(active_train_images, active_train_masks)))

while len(unlabeled_images) > 0:
    print ("Active acquition ", acquisitions, " in experiment num ", i)
    #use the initial list for the first unlabeled set
    if acquisitions == 0:
        unlabeled_dataframe = pd.DataFrame(list(zip(unlabeled_images, unlabeled_masks)))

    unlabeled_generator = get_generator_with_filename(unlabeled_dataframe, C)
    active_train_generator = get_generator(active_train_dataframe, C)

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


    network = model_instance(C)
    model = network.getModel()
    # # model.summary()
    if C.early_stop == 1:
        es = EarlyStopping(monitor='val_mean_iou', mode='auto', patience = C.early_stop_patience, verbose=1)
        history = model.fit_generator(generator = active_train_generator, steps_per_epoch=(len(active_train_dataframe)//C.batch_size) + 1, epochs=C.epochs, use_multiprocessing = True
                                  ,validation_data = val_generator, validation_steps = (len(Val_images)//C.batch_size)+1, callbacks=[es])
    else:
        history = model.fit_generator(generator = active_train_generator, steps_per_epoch=(len(active_train_dataframe)//C.batch_size) + 1, epochs=C.epochs, use_multiprocessing = True
                                      ,validation_data = val_generator, validation_steps = (len(Val_images)//C.batch_size)+1)

    #acquisition types


    unlabeled_scores_dict = dict()
    if acquisition_type == 1:
        # fully partial posterior query by dropout committee with KL
        print('Calculting entropy Acquisition')
        for it in range((len(unlabeled_dataframe)//C.batch_size) + 1):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = model.layers[-1].output.get_shape().as_list()[1:]
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
        for it in range((len(unlabeled_dataframe)//C.batch_size) + 1):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = model.layers[-1].output.get_shape().as_list()[1:]
            standard_prediction = model.predict(img)
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
        for it in range((len(unlabeled_dataframe)//C.batch_size) + 1):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = model.layers[-1].output.get_shape().as_list()[1:]
            standard_prediction = model.predict(img).astype(np.double)
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
        for it in range((len(unlabeled_dataframe)//C.batch_size) + 1):
            img, _, fn_imgs, _ = next(unlabeled_generator)
            out_shape = model.layers[-1].output.get_shape().as_list()[1:]
            standard_prediction = model.predict(img).astype(np.double)
            stochastic_predictions = np.zeros((img.shape[0], *out_shape))
            for dropout_i in range(C.dropout_iterations):
                stochastic_predictions += network.stochastic_foward_pass(img)
            stochastic_predictions /= C.dropout_iterations #average stochastic prediction
            kl_score = cynthonized_functions.score_jensen(standard_prediction, stochastic_predictions)
            for score_index, s in enumerate(kl_score):
                unlabeled_scores_dict[fn_imgs[score_index]] = s
        unlabeled_scores_dict = sorted(unlabeled_scores_dict.items(), key=lambda kv: kv[1])
        print('Done Calculating Commitee-KL Acquisition')




    print (unlabeled_scores_dict)
    exit()
    #collect logs
    training_loss[len(active_train_dataframe)] = history.history['loss']
    val_loss[len(active_train_dataframe)] = history.history['val_loss']
    training_dice_coef[len(active_train_dataframe)] = history.history['dice_coef']
    val_dice_coef[len(active_train_dataframe)] = history.history['val_dice_coef']

    print ("Evaluating mcmc metrics")
    mcmc_loss, mcmc_dice = network.stochastic_evaluate_generator(val_generator, C, (len(Val_images)//C.batch_size)+1)
    print ("Done evaluating mcmc metrics")
    # st_loss, st_dice = model.evaluate_generator(val_generator, steps=(len(Val_images)//C.batch_size)+1, use_multiprocessing=True)
    val_loss_mcmc.append(mcmc_loss)
    val_dice_coef_mcmc.append(mcmc_dice)

    print("Done with acquisition no. ", acquisitions, "out of ", len(unlabeled_dataframe)//C.active_batch, "remaining acquisitions in experiment number ", i)


    # select the most informative
    most_informative = unlabeled_dataframe.loc[unlabeled_dataframe[0].isin([i[0] for i in unlabeled_scores_dict[:C.active_batch]]) ]
    #add to the active training set
    active_train_dataframe = pd.concat([active_train_dataframe, most_informative])
    #remove from the unlabeled set

    unlabeled_dataframe = unlabeled_dataframe.loc[~unlabeled_dataframe[0].isin(most_informative[0])]

    #count acquisition rounds



    acquisitions += 1

    if acquisitions > 2:
        break



theArrayLog = np.array([(training_loss.items()),(val_loss.items()), (val_loss_mcmc), (training_dice_coef.items()), (val_dice_coef.items())
        ,(val_dice_coef_mcmc),(dataset_sizes_used), (C.num_classes)])

try:
    savedArrayLog = np.load("./results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(C.epochs)+"_"+str(acquisition_type)+".npy", allow_pickle = True)
    #append here
    savedArrayLog = np.vstack((savedArrayLog, theArrayLog))
    np.save("results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(C.epochs)+"_"+str(acquisition_type), savedArrayLog)

except IOError as e:
    np.save("results/"+log_filename+"_"+str(parser.dropout_type)+"_"+str(C.standard_dropout)+"_"+str(C.epochs)+"_"+str(acquisition_type), theArrayLog)


    # exit()
    #     index_maximum=bald(net_instance, unlabeled_X, C)
    # elif arg.q_type == 2: #random
    #     print ('Random acquisition')
    #     shuffled_indices = np.arange(unlabeled_X.shape[0])
    #     np.random.shuffle(shuffled_indices)
    #     index_maximum = shuffled_indices[:C.active_batch]
    # elif arg.q_type == 3:
    #     print ('Entropy with independent pixel assumptions')
    #     index_maximum = independent_pixel_entropy(net_instance, unlabeled_X, C)
    # else:
    #     exit()









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
