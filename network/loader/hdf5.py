from loader import Loader
import numpy as np
import h5py


class LoaderHdf5(Loader):
    def __init__(self, x):
        # Prepare streams and check each has the same number of samples
        self.data = [v for k, v in h5py.File(x).iteritems()]
        if len(np.unique([len(stream) for stream in self.data])) > 1:
            raise ValueError('The different streams specified in the h5py file have different number of samples!')

        # Initialize parent class
        Loader.__init__(self, len(self.data[0]))

    def _sample(self, indices):
        return tuple(stream[indices, ...] for stream in self.data)
