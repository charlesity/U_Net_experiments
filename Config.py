class Config:

    def __init__(self):
        # Set some parameters
        #Randomness parameters
        self.seed = 42
        self.n_experiments = 2
        self.epoch = 100
        self.batch_size = 32
        self.standard_dropout = 0.5
        self.max_pool_maxpool_dropout = 0.5



        # these parameters are not available until data is seen
        self.TRAIN_PATH = None
        self.TEST_PATH = None

        self.IMG_WIDTH = None
        self.IMG_HEIGHT = None
        self.IMG_CHANNELS = None

        #None Until argument is parsed
        self.enable_standard_dropout = None
        self.enable_maxpool_dropout = None

        self.IMG_WIDTH = 128
        self.IMG_HEIGHT = 128
        self.IMG_CHANNELS = 3




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
