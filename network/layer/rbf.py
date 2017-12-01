from layer import Layer

import tensorflow as tf


class LayerRbf(Layer):
    def __init__(self, shape):
        self.shape = shape
        self.input_shape = (self.shape[0],)
        Layer.__init__(self, True, [
            {'name': 'w',  'type': 'w', 'shape': shape},
            {'name': 's',  'type': 'w', 'shape': (1, shape[-1])},
        ])

    def forward(self, x):
        distances = tf.reduce_sum((tf.expand_dims(x, axis=2) - tf.expand_dims(self.w, axis=0)) ** 2, axis=1)
        return tf.exp(-distances / (2 * (self.s ** 2)))

    def get_input_shape(self):
        return [self.shape[0]]

    def get_output_shape(self):
        return [self.shape[-1]]
