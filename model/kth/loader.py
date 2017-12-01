from network.loader import Loader
import numpy as np
from random import shuffle
import h5py


# ------------------------------------------------------------------
# DEFINE CUSTOM DATA LOADER
# ------------------------------------------------------------------

class LoaderKth(Loader):
    def __init__(self, x, sample_length=20, sample_frequency=1, shuffle_data=True):
        self.s_len = sample_length
        self.s_freq = sample_frequency
        self.s_window = (self.s_len - 1) * self.s_freq + 1

        # Prepare reader for each video sequence
        self.data = [v for k, v in h5py.File(x, 'r').iteritems()]
        self.lengths = [len(v) for v in self.data]

        # Remove videos with sequence length smaller than sample_length
        self.data, self.lengths = zip(*[(v, l) for v, l in zip(self.data, self.lengths) if l >= self.s_window])

        # Prepare mapping (sample to sequence)
        self.mapping = np.concatenate(tuple(
            np.array([i] * (l - self.s_window + 1)) for i, l in enumerate(self.lengths)
        ), axis=0)

        # Prepare padding (sample to sequence padding)
        self.padding = np.concatenate(tuple(
            np.arange(0, l - self.s_window + 1) for i, l in enumerate(self.lengths)
        ), axis=0)

        # Call parent constructor
        Loader.__init__(self, len(self.mapping), shuffle_data=shuffle_data)

    def _sample(self, indices):
        s_pos = [(self.data[self.mapping[i]], self.padding[i]) for i in indices]
        ret = np.stack(tuple(s[off:off+self.s_window:self.s_freq] for s, off in s_pos), axis=0)
        return np.cast[np.float32](ret) / 255
