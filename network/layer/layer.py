from ..toolkit import init_biases, init_weights
import tensorflow as tf
import numpy as np


class Layer:
    def __init__(self, computable, params):
        inits = {
            'b': init_biases,
            'w': init_weights
        }

        # Build layer parameters
        self.computable = computable
        self.__dict__.update({p['name']: inits[p['type']](p['shape']) for p in params})

    def forward(self, x):
        raise NotImplementedError('Forward pass not implemented for this layer!')

    def backward(self, x):
        raise NotImplementedError('Backward pass not implemented for this layer!')

    def is_computable(self):
        return self.computable

    def get_input_shape(self):
        raise NotImplementedError(
            'get_input_shape not implemented for this layer!'
        )

    def get_output_shape(self):
        raise NotImplementedError(
            'get_output_shape not implemented for this layer!'
        )

    # DEFINE PARAMETER GATHERING METHODS
    # -------------------------------------------------------------------

    def gather(self, placeholder=False):
        # If no gather function, return empty list
        if not hasattr(self, '_gather'):
            return []

        # Gather parameters, format into list
        ret = getattr(self, '_gather')(placeholder)
        return ret if isinstance(ret, list) else [ret]

    def scatter(self, values):
        # If no scatter function, return
        if not hasattr(self, '_scatter'):
            return

        # Scatter parameter values
        getattr(self, '_scatter')(values)

