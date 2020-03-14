
from keras.models import Model, load_model
from keras.layers import Input, Activation, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
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

        encoder_filter_list = [64, 128, 256, 512, 1024]
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
            x = Dropout(C.standard_dropout) (c1) if C.enable_standard_dropout else (x)
            x = (Activation('relu'))(x)
            x = (BatchNormalization())(x)
            x = (MaxPooling2D((pool_size, pool_size), data_format=C.IMAGE_ORDERING))(x)
            x = Dropout(C.max_pool_maxpool_dropout) (x) if C.enable_maxpool_dropout else (x)
            levels.append(x)

        [f1, f2, f3, f4, f5] = levels
        #decoders/image reconstruction layers
        # ground level
        o = f5
        o = (Conv2D(1024, (3, 3), padding='same', data_format=C.IMAGE_ORDERING))(o)
        o = Dropout(C.standard_dropout) (o) if C.enable_standard_dropout else (o)
        o = (Activation('relu'))(o)
        o = (BatchNormalization())(o)

        o = (Conv2D(1024, (3, 3), padding='same', data_format=C.IMAGE_ORDERING))(o)
        o = Dropout(C.standard_dropout) (o) if C.enable_standard_dropout else (o)
        o = (Activation('relu'))(o)
        o = (BatchNormalization())(o)

        #image reconstruction
        for index, d_filter in enumerate(decoder_filter_lists):
            o = (UpSampling2D((2, 2), data_format=C.IMAGE_ORDERING))(o)
            o = (concatenate([o, f4], axis=MERGE_AXIS))
            o = (Conv2D(d_filter, (3, 3), padding='same', data_format=C.IMAGE_ORDERING))(o)
            o = Dropout(C.standard_dropout) (o) if C.enable_standard_dropout else (o)
            o = (BatchNormalization())(o)

            o = (Conv2D(d_filter, (3, 3), padding='same', data_format=C.IMAGE_ORDERING))(o)
            o = Dropout(C.standard_dropout) (o) if C.enable_standard_dropout else (o)
            o = (BatchNormalization())(o)

        o = Conv2D(n_classes, (3, 3), padding='same', data_format=C.IMAGE_ORDERING)(o)
        self.model = Model(inputs=[inputs], outputs=[o])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[self.mean_iou])

        #function for sampling committee member via forward passes
        self.f = K.function([self.model.layers[0].input, K.learning_phase()],[self.model.layers[-1].output])

        es = EarlyStopping(monitor='val_mean_iou', mode='min', verbose=1)

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

    def mean_iou(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.9, 1.0, 0.5):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)
