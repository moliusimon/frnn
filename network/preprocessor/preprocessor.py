from operator import build_operator
from operator.op_swapaxes import OperatorSwapaxes
from ..loader import build_loader
import tensorflow as tf
import numpy as np


class Preprocessor:
    def __init__(self, operators, loader=None):
        # Build preprocessor operators
        self.operators = [] if operators is None else [build_operator(op) for op in operators]

        # If loader provided, prepare number of streams and placeholders
        self.loader, self.placeholders = None, []
        self.__init_loader__(loader)

    def __init_loader__(self, loader):
        self.loader = build_loader(loader)
        self.placeholders = []

        # Make stub sampling and set number of streams
        temp_streams = self.sample(1)
        self.loader.reset()
        self.num_streams = len(temp_streams)

        # Build placeholders
        for stream, b_index in zip(temp_streams, self._get_batch_indices()):
            shape = stream.shape[:b_index] + (None,) + stream.shape[b_index + 1:]
            dtype = tf.float32 if np.issubdtype(stream.dtype, np.float) else tf.int64
            self.placeholders.append(tf.placeholder(dtype, shape=shape))

    def get_placeholders(self):
        return self.placeholders

    def has_batch(self):
        return self.loader.has_batch()

    def set_loader(self, loader):
        self.__init_loader__(loader)

    def sample(self, batch_size, mode='train', full_augment=False):
        # Check loader is provided
        if self.loader is None:
            raise ValueError('Cannot directly sample from the preprocessor: Data loader not provided!')

        streams = self.loader.sample(mode, batch_size)
        return self.preprocess(streams, full_augment)

    def retrieve(self, indices=None):
        # Check loader is provided
        if self.loader is None:
            raise ValueError('Cannot retrieve data from the preprocessor: Data loader not provided!')

        streams = self.loader.retrieve(indices)
        return self.preprocess(streams, full_augment=False)

    def preprocess(self, streams, full_augment=False):
        for op in self.operators:
            streams = op.apply(streams, full_augment)
        return streams

    def _get_batch_indices(self):
        indices = [0] * self.num_streams
        for op in self.operators:
            if isinstance(op, OperatorSwapaxes):
                indices = [np.argwhere(np.array(op.order) == indices[i])[0][0] for i in indices]

        return indices
