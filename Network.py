
from keras.models import Model, load_model
from keras.layers import Input, Activation, UpSampling2D, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from sklearn.metrics import log_loss
from keras import backend as K
K.clear_session()

import tensorflow as tf

import numpy as np





class Network():
    def __init__(self, C):

        encoder_filter_list = [64, 128, 256, 512]
        decoder_filter_lists = [512, 256, 128, 24]
        x = C.img_input
        levels = []
        # vanilla encoders
        for e_filter in encoder_filter_list:
            x = (Conv2D(e_filter, (C.kernel_size, C.kernel_size), data_format=C.IMAGE_ORDERING, padding='same'))(x)
            x = Dropout(C.standard_dropout) (x) if C.enable_standard_dropout else (x)
            x = (BatchNormalization())(x)
            x = (Activation('relu'))(x)

            x = (Conv2D(e_filter, (C.kernel_size, C.kernel_size), data_format=C.IMAGE_ORDERING, padding='same'))(x)
            x = Dropout(C.standard_dropout) (x) if C.enable_standard_dropout else (x)
            x = (BatchNormalization())(x)
            x = (Activation('relu'))(x)

            levels.append(x)
            x = (MaxPooling2D((C.pool_size, C.pool_size), data_format=C.IMAGE_ORDERING))(x)
            x = Dropout(C.max_pool_maxpool_dropout) (x) if C.enable_maxpool_dropout else (x)

        [f1, f2, f3, f4] = levels
        # ground level
        o = f4
        o = x
        o = (Conv2D(1024, (C.kernel_size, C.kernel_size), padding='same', data_format=C.IMAGE_ORDERING))(o)
        o = Dropout(C.standard_dropout) (o) if C.enable_standard_dropout else (o)
        o = (Activation('relu'))(o)
        o = (BatchNormalization())(o)


        o = (Conv2D(1024, (C.kernel_size, C.kernel_size), padding='same', data_format=C.IMAGE_ORDERING))(o)
        o = Dropout(C.standard_dropout) (o) if C.enable_standard_dropout else (o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)


        #image reconstruction
        for index, d_filter in enumerate(decoder_filter_lists):
            o = (UpSampling2D((C.uppool_size, C.uppool_size), data_format=C.IMAGE_ORDERING))(o)
            o = (concatenate([o, levels[-(index+1)]], axis=C.MERGE_AXIS))

            o = (Conv2D(d_filter, (C.kernel_size, C.kernel_size), padding='same', data_format=C.IMAGE_ORDERING))(o)
            o = Dropout(C.standard_dropout) (o) if C.enable_standard_dropout else (o)
            o = (BatchNormalization())(o)

            o = (Conv2D(d_filter, (C.kernel_size, C.kernel_size), padding='same', data_format=C.IMAGE_ORDERING))(o)
            o = Dropout(C.standard_dropout) (o) if C.enable_standard_dropout else (o)
            o = (BatchNormalization())(o)

        o = Conv2D(C.num_classes, (C.kernel_size, C.kernel_size), padding='same', data_format=C.IMAGE_ORDERING)(o)
        o = (Activation('softmax'))(o)
        self.model = Model(inputs=[C.img_input], outputs=[o])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[self.dice_coef])

        #function for sampling committee member via forward passes
        self.f = K.function([self.model.layers[0].input, K.learning_phase()],[self.model.layers[-1].output])

        es = EarlyStopping(monitor='dice_coef', mode='min', verbose=1)

    def getModel(self):
        return self.model

    def stochastic_predict(self, test_imgs, C):
        out_shape = self.model.layers[-1].output.get_shape().as_list()[1:]
        pred_mask = np.zeros(shape = (test_imgs.shape[0], *out_shape), dtype=float)
        for i in range(C.dropout_iterations):
            pred_mask = pred_mask + self.f((test_imgs, 1))[0]
        print(pred_mask[0,0,0,:])
        pred_mask = pred_mask/C.dropout_iterations
        return pred_mask

    def stochastic_evaluate_generator(self, generator, C, steps=None):
        loss_all = np.array([])
        dice_coef = np.array([])
        for i in range(steps):
            img, mask = next(generator)
            mask = mask.reshape(mask.shape[0], mask.shape[1]*mask.shape[2], mask.shape[3])
            l= np.zeros(shape=(img.shape[0]))
            d = np.zeros(shape=(img.shape[0]))
            for i in range(C.dropout_iterations):
                pred = self.f((img, 1))[0]
                pred = pred.reshape(pred.shape[0], pred.shape[1]*pred.shape[2], pred.shape[3])
                d += np.array([self.dice_coef_cpu(mask[j], pred[j]) for j in range(pred.shape[0])])
                l += np.array([log_loss(mask[i], pred[i]) for i in range(pred.shape[0])])
            d /= C.dropout_iterations
            l /= C.dropout_iterations
            dice_coef = np.append(dice_coef, d)
            loss_all = np.append(loss_all, l)
        return np.mean(loss_all), np.mean(dice_coef)

    def dice_coef_cpu(self, y_true, y_pred, smooth=1e-12):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f*y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f)
        return  (2. * intersection + smooth)/(union + smooth)

    def stochastic_foward_pass(self, x_unlabeled):
        return self.f((x_unlabeled, 1))[0]

    def dice_coef(self, y_true, y_pred, smooth=1e-12):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
