from layer import Layer

import tensorflow as tf


class LayerMarginalize(Layer):
    def __init__(self, shape):
        self.shape = shape
        Layer.__init__(self, False, [
            {'name': '_marginal_init', 'type': 'b', 'shape': shape[1:]}
        ])

        # Define state placeholder
        self.marginal = tf.placeholder(tf.float32, [None, self.shape[-1]])
        self._marginal = self.marginal

    def forward(self, x):
        x_shape = [v.value for v in x.get_shape()]
        splits = tf.split(x, [x_shape[-1]-self.shape[-1], self.shape[-1]], len(x_shape)-1)
        self._marginal = splits[1]
        return splits[0]

    def backward(self, x):
        return tf.concat([x, self._marginal], -1)

    def _gather(self, placeholder=False):
        return [(self.marginal, self._marginal_init) if placeholder else self._marginal]

    def _scatter(self, values):
        self._marginal = values.pop(0)
