from utilities import *
import cv2 as cv
import pandas as pd
import glob as glob
import argparse
from Config import Config
from PIL import Image

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-ds", "--dataset_location",  required = True, help ="link to the datasource")
parser.add_argument("-ag_n", "--augment_number",  required = True, help ="Augment by how many times")


parser = parser.parse_args()

data_gen_args = dict(rotation_range= 90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip = True,
                     vertical_flip = True,
                     zoom_range=0.2)
dataset_location = parser.dataset_location
ag_n = int(parser.augment_number)
all_images = glob.glob(dataset_location+'/image/*[0-9].png')
all_masks = glob.glob(dataset_location+'/label/*[0-9].png')


dataset = pd.DataFrame(list(zip(all_images, all_masks)))

C=Config()
C.batch_size = 4

sample_image = cv.imread(all_images[0])
# sample_mask =cv.imread(all_masks[0])
#
# print (np.unique(sample_mask), sample_mask.shape)
# plt.imshow(sample_mask)
# plt.show()
# exit()
C.IMG_WIDTH = sample_image.shape[0]
C.IMG_HEIGHT = sample_image.shape[1]
C.IMG_CHANNELS = sample_image.shape[2]

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


image_generator  = image_datagen.flow_from_dataframe(dataset, target_size=(C.IMG_WIDTH, C.IMG_HEIGHT), x_col=0, batch_size=C.batch_size,  class_mode=None, seed= C.randomSeed)
mask_generator = image_datagen.flow_from_dataframe(dataset, target_size=(C.IMG_WIDTH, C.IMG_HEIGHT), color_mode='grayscale', x_col=1, batch_size=C.batch_size, class_mode=None, seed= C.randomSeed)

generator = zip(image_generator, mask_generator)
aug_counter = 0
#augment the dataset 4 times
for i in range ((len(all_images)//C.batch_size) * ag_n):
    img, mask = next(generator)

    # fig, ax = plt.subplots(img.shape[0],2)
    for j in range(img.shape[0]):
        try:
            im = mask[j].reshape(mask[j].shape[0], mask[j].shape[1]).astype(np.uint8)
            cv.imwrite(dataset_location+"/image/"+"aug"+str(aug_counter)+".png",img[j])
            cv.imwrite(dataset_location+"/label/"+"aug"+str(aug_counter)+".png",im)

        except Exception as e:
            print (e)
        print (np.unique(im), im.shape)
        # ax[j][0].imshow(img[j].astype(np.uint8))
        # ax[j][1].imshow(mask[j].reshape(mask[j].shape[0], mask[j].shape[1]).astype(np.uint8))


        aug_counter += 1
    # plt.show()


test_img = cv.imread(dataset_location+"/label/"+"aug"+str(aug_counter-1)+".png")
# test_img = cv.imread(dataset_location+"/label/"+"aug"+str(100)+".png")
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
print(np.unique(test_img), test_img.shape)
plt.imshow(test_img.astype(np.uint8))
plt.show()
