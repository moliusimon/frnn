from network.loader import Loader
import numpy as np
import h5py


class LoaderMMNIST(Loader):
    def __init__(self, x):
        # Prepare instances
        self.data = [v for k, v in h5py.File(x, 'r').iteritems()]

        # Initialize parent class
        Loader.__init__(self, len(self.data))

    def _sample(self, indices):
        return np.cast[np.float32](np.stack([self.data[i][...] for i in indices], axis=0)) / 255
