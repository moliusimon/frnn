from layer import Layer

import tensorflow as tf


class LayerFully(Layer):
    def __init__(self, shape, activation=tf.nn.tanh):
        self.shape = shape
        self.input_shape = (self.shape[0],)
        self.activation = (lambda x: x) if activation is None else activation
        Layer.__init__(self, True, [
            {'name': 'b1', 'type': 'b', 'shape': shape[-1]},
            {'name': 'b2', 'type': 'b', 'shape': shape[-2]},
            {'name': 'w',  'type': 'w', 'shape': shape}
        ])

    def forward(self, x):
        return self.activation(tf.nn.bias_add(tf.matmul(x, self.w), self.b1))

    def backward(self, x):
        w = tf.transpose(self.w)
        return self.activation(tf.nn.bias_add(tf.matmul(x, w), self.b2))

    def get_input_shape(self):
        return [self.shape[0]]

    def get_output_shape(self):
        return [self.shape[-1]]
