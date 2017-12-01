from loader import Loader
import numpy as np


class LoaderMultistream(Loader):
    def __init__(self, x):
        # Check all streams have the same number of samples
        self.streams = x
        if len(np.unique([stream.num_samples for stream in self.streams])) > 1:
            raise ValueError('Multi-stream loader has different number of samples on each stream!')

        # Call parent constructor
        Loader.__init__(self, self.streams[0].num_samples)

    def _reset(self):
        for s in self.streams:
            s.counter = 0

    def _sample(self, indices):
        return tuple(stream._sample(indices) for stream in self.streams)
