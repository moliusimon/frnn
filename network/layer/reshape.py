from layer import Layer

import tensorflow as tf


class LayerReshape(Layer):
    def __init__(self, shape_in, shape_out):
        self.shape_in = [-1]+list(shape_in)
        self.shape_out = [-1]+list(shape_out)
        Layer.__init__(self, False, [])

    def forward(self, x):
        return tf.reshape(x, self.shape_out)

    def backward(self, x):
        return tf.reshape(x, self.shape_in)

    def get_input_shape(self):
        return self.shape_in

    def get_output_shape(self):
        return self.shape_out