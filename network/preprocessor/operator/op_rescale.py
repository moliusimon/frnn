from operator import Operator
import numpy as np


class OperatorRescale(Operator):
    def __init__(self, weight=1, bias=0):
        self.weight = weight
        self.bias = bias
        Operator.__init__(self, num_modes=1)

    def apply_random(self, streams):
        return self.apply_full(streams)

    def apply_full(self, streams):
        return tuple(self.weight * stream + self.bias for stream in streams)
