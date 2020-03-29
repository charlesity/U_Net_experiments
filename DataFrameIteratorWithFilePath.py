import numpy as np
from keras_preprocessing.image import DataFrameIterator

class DataFrameIteratorWithFilePath(DataFrameIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.filenames_np = np.array(self.filepaths)


    def _get_batches_of_transformed_samples(self, index_array):
        return (super()._get_batches_of_transformed_samples(index_array),
                self.filenames_np[index_array])
