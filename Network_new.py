
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import losses as losses

K.clear_session()

import tensorflow as tf

import numpy as np



class Network():
    def __init__(self, C):
        self.C = C

        if K.image_data_format() == 'channels_first':
            input_shape = (C.IMG_CHANNELS, C.IMG_HEIGHT, C.IMG_WIDTH)
        else:
            input_shape = (C.IMG_HEIGHT, C.IMG_WIDTH, C.IMG_CHANNELS)

        inputs = Input(shape=input_shape)

        s = Lambda(lambda x: x / 255) (inputs)

        if C.regularizers:
            c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'
                        ,kernel_regularizer=regularizers.l2(0.01)) (s)

            c1 = Dropout(C.standard_dropout) (c1) if C.enable_standard_dropout else c1

            c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                        kernel_regularizer=regularizers.l2(0.01)) (c1)

            c1 = Dropout(C.max_pool_maxpool_dropout) (c1) if C.enable_maxpool_dropout else c1

            p1 = MaxPooling2D((2, 2)) (c1)

            c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (p1)

            c2 = Dropout(C.standard_dropout) (c2) if C.enable_standard_dropout else c2

            c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (c2)

            c2 = Dropout(C.max_pool_maxpool_dropout) (c2) if C.enable_maxpool_dropout else c2

            p2 = MaxPooling2D((2, 2)) (c2)

            c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (p2)

            c3 = Dropout(C.standard_dropout) (c3) if C.enable_standard_dropout else c3

            c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (c3)

            c3 = Dropout(C.max_pool_maxpool_dropout) (c3) if C.enable_maxpool_dropout else c3

            p3 = MaxPooling2D((2, 2)) (c3)

            c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (p3)

            c4 = Dropout(C.standard_dropout) (c4) if C.enable_standard_dropout else c4

            c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (c4)

            c4 = Dropout(C.max_pool_maxpool_dropout) (c4) if C.enable_maxpool_dropout else c4

            p4 = MaxPooling2D(pool_size=(2, 2)) (c4)


            c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (p4)

            c5 = Dropout(C.standard_dropout) (c5) if C.enable_standard_dropout else c5

            c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (c5)

            c5 = Dropout(C.max_pool_maxpool_dropout) (c5) if C.enable_maxpool_dropout else c5


            u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

            u6 = concatenate([u6, c4])

            c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (u6)

            c6 = Dropout(C.standard_dropout) (c6) if C.enable_standard_dropout else c6

            c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (c6)

            c6 = Dropout(C.max_pool_maxpool_dropout) (c6) if C.enable_maxpool_dropout else c6

            u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

            u7 = concatenate([u7, c3])

            c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same',kernel_regularizer=regularizers.l2(0.01)) (u7)

            c7 = Dropout(C.standard_dropout) (c7) if C.enable_standard_dropout else c7

            c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same', kernel_regularizer=regularizers.l2(0.01)) (c7)

            c7 = Dropout(C.max_pool_maxpool_dropout) (c7) if C.enable_maxpool_dropout else c7

            u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
            u8 = concatenate([u8, c2])

            c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                        padding='same',kernel_regularizer=regularizers.l2(0.01)) (u8)

            c8 = Dropout(C.standard_dropout) (c8) if C.enable_standard_dropout else c8

            c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'
                        ,padding='same', kernel_regularizer=regularizers.l2(0.01)) (c8)

            c8 = Dropout(C.max_pool_maxpool_dropout) (c8) if C.enable_maxpool_dropout else c8

            u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

            u9 = concatenate([u9, c1], axis=3)

            c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal'
                        ,padding='same', kernel_regularizer=regularizers.l2(0.01)) (u9)

            c9 = Dropout(C.standard_dropout) (c9) if C.enable_standard_dropout else c9

            c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal'
                        ,padding='same', kernel_regularizer=regularizers.l2(0.01)) (c9)

        else:

            c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)

            c1 = Dropout(C.standard_dropout) (c1) if C.enable_standard_dropout else c1

            c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)

            c1 = Dropout(C.max_pool_maxpool_dropout) (c1) if C.enable_maxpool_dropout else c1

            p1 = MaxPooling2D((2, 2)) (c1)

            c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (p1)
            c2 = Dropout(C.standard_dropout) (c2) if C.enable_standard_dropout else c2

            c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (c2)
            c2 = Dropout(C.max_pool_maxpool_dropout) (c2) if C.enable_maxpool_dropout else c2
            p2 = MaxPooling2D((2, 2)) (c2)

            c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (p2)
            c3 = Dropout(C.standard_dropout) (c3) if C.enable_standard_dropout else c3

            c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (c3)
            c3 = Dropout(C.max_pool_maxpool_dropout) (c3) if C.enable_maxpool_dropout else c3
            p3 = MaxPooling2D((2, 2)) (c3)

            c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (p3)
            c4 = Dropout(C.standard_dropout) (c4) if C.enable_standard_dropout else c4
            c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (c4)
            c4 = Dropout(C.max_pool_maxpool_dropout) (c4) if C.enable_maxpool_dropout else c4
            p4 = MaxPooling2D(pool_size=(2, 2)) (c4)


            c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (p4)
            c5 = Dropout(C.standard_dropout) (c5) if C.enable_standard_dropout else c5
            c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (c5)
            c5 = Dropout(C.max_pool_maxpool_dropout) (c5) if C.enable_maxpool_dropout else c5


            u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
            u6 = concatenate([u6, c4])
            c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (u6)
            c6 = Dropout(C.standard_dropout) (c6) if C.enable_standard_dropout else c6
            c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
            c6 = Dropout(C.max_pool_maxpool_dropout) (c6) if C.enable_maxpool_dropout else c6

            u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
            u7 = concatenate([u7, c3])
            c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (u7)
            c7 = Dropout(C.standard_dropout) (c7) if C.enable_standard_dropout else c7
            c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'
                        , padding='same') (c7)
            c7 = Dropout(C.max_pool_maxpool_dropout) (c7) if C.enable_maxpool_dropout else c7

            u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
            u8 = concatenate([u8, c2])
            c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'
                        ) (u8)
            c8 = Dropout(C.standard_dropout) (c8) if C.enable_standard_dropout else c8
            c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'
                        ) (c8)
            c8 = Dropout(C.max_pool_maxpool_dropout) (c8) if C.enable_maxpool_dropout else c8

            u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
            u9 = concatenate([u9, c1], axis=3)
            c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'
                        ) (u9)
            c9 = Dropout(C.standard_dropout) (c9) if C.enable_standard_dropout else c9
            c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'
                        ) (c9)

        outputs = Conv2D(C.num_classes, (1, 1), activation='sigmoid') (c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        # self.model.compile(optimizer='adam', loss=self.dice_coef_loss, metrics=[self.dice_coef])
        self.model.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=[self.dice_coef])

        #function for sampling committee member via forward passes
        self.f = K.function([self.model.layers[0].input, K.learning_phase()],[self.model.layers[-1].output])


    def getModel(self):
        return self.model

    def stochastic_predict(self, X_test, C):
        shape = np.append([C.dropout_iterations, X_test.shape[0]], C.output_shape)
        prediction_score = np.zeros(shape)
        for i in range(C.dropout_iterations):
            prediction_score[i] = self.f((X_test, 1))[0]
        return prediction_score.mean(axis = 0), prediction_score.std(axis = 0)

    def stochastic_predict_generator(self, generator, steps, C):
        out_prediction_mean = []
        out_prediction_std = []
        y_values = []   # it is more efficient to retreive the actual y here
        for i in range(steps):
            X, y = next(generator)
            shape = np.append([C.dropout_iterations, X.shape[0]], C.output_shape)
            prediction_score = np.zeros(shape)
            for i in range(C.dropout_iterations):
                prediction_score[i] = self.f((X, 1))[0]
            out_prediction_mean.append(prediction_score.mean(axis=0))
            out_prediction_std.append(prediction_score.std(axis=0))
            y_values.append(y)

        out_prediction_mean = np.concatenate(out_prediction_mean, axis = 0)
        out_prediction_std = np.concatenate(out_prediction_std, axis = 0)

        y_values = np.concatenate(y_values)
        return out_prediction_mean, out_prediction_std, y_values

    def stochastic_foward_pass(self, x_unlabeled):
        return self.f((x_unlabeled, 1))[0]


    def dice_coef_loss(self,y_true, y_pred):
        return 1.-self.dice_coef(y_true, y_pred)


    def dice_coef(self, y_true, y_pred):
        smooth = 1e-5
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

    def dice_coef_old(self, y_true, y_pred):
        smooth = 1e-5
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)
