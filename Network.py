
from keras.models import Model, load_model
from keras.layers import Input, Activation, BatchNormalization, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.callbacks import EarlyStopping
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
            x = (Activation('relu'))(x)
            x = (BatchNormalization())(x)

            x = (Conv2D(e_filter, (C.kernel_size, C.kernel_size), data_format=C.IMAGE_ORDERING, padding='same'))(x)
            x = Dropout(C.standard_dropout) (x) if C.enable_standard_dropout else (x)
            x = (Activation('relu'))(x)
            x = (BatchNormalization())(x)
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
        o = (Activation('relu'))(o)
        o = (BatchNormalization())(o)

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

    def stochastic_predict(self, X_test, C):
        shape = np.append([C.dropout_iterations, X_test.shape[0]], C.output_shape)
        prediction_score = np.zeros(shape)
        for i in range(C.dropout_iterations):
            prediction_score[i] = self.f((X_test, 1))[0]
        return prediction_score.mean(axis = 0)

    def stochastic_foward_pass(self, x_unlabeled):
        return self.f((x_unlabeled, 1))[0]

    def dice_coef(self, y_true, y_pred, smooth=1e-12):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
