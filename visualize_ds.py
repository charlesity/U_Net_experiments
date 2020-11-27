#Charles I. Saidu

import numpy as np
import glob as glob

import cv2
import matplotlib.pyplot as plt
from utilities import *


dataset = [("/membrane/", "png", 2), ("/DIC_C2DH_Hela/","png", 20), ("/PhC-C2DH-U373/", "png", 14), ("/warwick/", "bmp", 50)]

ds1_images = glob.glob("./dataset"+dataset[0][0]+'/dataset/image/*.'+dataset[0][1])
ds1_masks = glob.glob("./dataset"+dataset[0][0]+'/dataset/label/*.'+dataset[0][1])

ds2_images = glob.glob("./dataset"+dataset[1][0]+'/dataset/image/*.'+dataset[1][1])
ds2_masks = glob.glob("./dataset"+dataset[1][0]+'/dataset/label/*.'+dataset[1][1])

ds3_images = glob.glob("./dataset"+dataset[2][0]+'/dataset/image/*.'+dataset[2][1])
ds3_masks = glob.glob("./dataset"+dataset[2][0]+'/dataset/label/*.'+dataset[2][1])

ds4_images = glob.glob("./dataset"+dataset[3][0]+'/dataset/image/*.'+dataset[3][1])
ds4_masks = glob.glob("./dataset"+dataset[3][0]+'/dataset/label/*.'+dataset[3][1])




img_1 = cv2.imread(ds1_images[4],cv2.IMREAD_COLOR)
img_1_mask = cv2.imread(ds1_masks[4],cv2.IMREAD_UNCHANGED)

img_2 = cv2.imread(ds2_images[11],cv2.IMREAD_COLOR)
img_2_mask = cv2.imread(ds2_masks[11],cv2.IMREAD_UNCHANGED)

img_3 = cv2.imread(ds3_images[9],cv2.IMREAD_COLOR)
img_3_mask = cv2.imread(ds3_masks[9],cv2.IMREAD_UNCHANGED)

img_4 = cv2.imread(ds4_images[42],cv2.IMREAD_COLOR)
img_4_mask = cv2.imread(ds4_masks[42],cv2.IMREAD_UNCHANGED)


# f, axes = plt.subplots(2, 4)


# axes[0,0].set_yticklabels([])
# axes[0,0].set_xticklabels([])
# axes[0,0].imshow(img_1)

# axes[0,1].imshow(img_1_mask)
# axes[0,1].set_yticklabels([])
# axes[0,1].set_xticklabels([])

# axes[0,2].imshow(img_2)
# axes[0,2].set_yticklabels([])
# axes[0,2].set_xticklabels([])

# axes[0,3].imshow(img_2_mask)
# axes[0,3].set_yticklabels([])
# axes[0,3].set_xticklabels([])

# axes[1,0].imshow(img_3)
# axes[1,0].set_yticklabels([])
# axes[1,0].set_xticklabels([])

# axes[1,1].imshow(img_3_mask)
# axes[1,1].set_yticklabels([])
# axes[1,1].set_xticklabels([])

# axes[1,2].imshow(img_4)
# axes[1,2].set_yticklabels([])
# axes[1,2].set_xticklabels([])

# axes[1,3].imshow(img_4_mask)
# axes[1,3].set_yticklabels([])
# axes[1,0].set_xticklabels([])

# plt.show()

plt.figure()
plt.imshow(img_1)
plt.xticks([])
plt.yticks([])


plt.figure()
plt.imshow(img_1_mask)
plt.xticks([])
plt.yticks([])


plt.figure()
plt.imshow(img_2)
plt.xticks([])
plt.yticks([])


plt.figure()
plt.imshow(img_2_mask)
plt.xticks([])
plt.yticks([])


plt.figure()
plt.imshow(img_3)
plt.xticks([])
plt.yticks([])


plt.figure()
plt.imshow(img_3_mask)
plt.xticks([])
plt.yticks([])


plt.figure()
plt.imshow(img_4)
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(img_4_mask)

plt.xticks([])
plt.yticks([])

plt.show()
