from keras.layers import Input
from keras import backend as K

class Config:

    def __init__(self, IMG_WIDTH = 64, IMG_HEIGHT = 64, IMG_CHANNELS = 3):
        # Set some parameters
        #Randomness parameters
        self.seed = 42
        self.n_experiments = 10
        self.epochs = 200
        self.batch_size = 4
        self.standard_dropout = 0.5
        self.max_pool_maxpool_dropout = 0.5
        self.randomSeed = 1
        self.kernel_size = 3
        self.pool_size = 2
        self.uppool_size = 2
        self.num_classes = 2


        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        self.IMAGE_ORDERING = None


        # these parameters are not available until data is seen

        self.num_classes = None
        self.TRAIN_PATH = None
        self.TEST_PATH = None


        #None Until argument is parsed
        self.enable_standard_dropout = None
        self.enable_maxpool_dropout = None
        self.output_shape = None

        self.steps_per_epoch_scale = 1
        self.initial_training_ratio = .01
        self.num_active_queries = 15  # 15  #total number of active learning queries
        self.subsample_size = 100
        self.dropout_iterations = 50 # 50
        self.active_batch = 10

        self.mask_test_set_ratio = .35
        self.mask_threshold = .6


        self.early_stop = 1
        self.early_stop_patience = 25 # 10 percent of epoch
        self.regularizers = True


        if K.image_data_format() == 'channels_first':
            self.input_shape = (self.IMG_CHANNELS, self.IMG_HEIGHT, self.IMG_WIDTH)
            self.IMAGE_ORDERING = 'channels_first'
            self.MERGE_AXIS = 1
        else:
            self.input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)
            self.IMAGE_ORDERING = 'channels_last'
            self.MERGE_AXIS = - 1

        self.img_input = Input(shape=self.input_shape)


    def set_enable_dropout(self, argument):
        if argument == 0:
            self.enable_standard_dropout = False
            self.enable_maxpool_dropout = False

        elif argument == 1:
            self.enable_standard_dropout = True
            self.enable_maxpool_dropout = False

        elif argument == 2:
            self.enable_standard_dropout = False
            self.enable_maxpool_dropout = True
        else:
            self.enable_standard_dropout = True
            self.enable_maxpool_dropout = True
