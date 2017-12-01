from operator import Operator
import numpy as np


class OperatorSwapaxes(Operator):
    def __init__(self, order):
        self.order = tuple(order)
        Operator.__init__(self, num_modes=1)

    def apply_random(self, streams):
        return self.apply_full(streams)

    def apply_full(self, streams):
        return tuple(np.transpose(stream, axes=self.order) for stream in streams)
