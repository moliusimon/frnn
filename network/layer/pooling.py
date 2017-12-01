from layer import Layer

import tensorflow as tf
import numpy as np


class LayerPooling(Layer):
    def __init__(self, pool_size, reverted=False):
        self.pool_size = list(pool_size)
        self.reverted = reverted
        Layer.__init__(self, False, [])

    def forward(self, x):
        return self._backward(x) if self.reverted else self._forward(x)

    def backward(self, x):
        return self._forward(x) if self.reverted else self._backward(x)

    def _forward(self, x):
        return tf.nn.pool(x, self.pool_size, 'MAX', padding='SAME', strides=self.pool_size)

    def _backward(self, x):
        ts = [v.value for v in x.get_shape()]
        shape_t = [p * v for p, v in zip(ts[1:-1], self.pool_size)]

        # Replicate values and reshape replica dimension into pool shape
        x = tf.stack([x for _ in range(np.prod(self.pool_size))], axis=len(self.pool_size)+1)
        x = tf.reshape(x, [-1]+ts[1:-1]+self.pool_size+[ts[-1]])

        # Reorder dimensions and reshape array to its interpolated size
        ordering = list(np.transpose(np.reshape(np.arange(1, 2*len(shape_t)+1), [2, len(shape_t)])).flatten())
        return tf.reshape(tf.transpose(x, perm=[0]+ordering+[2*len(self.pool_size)+1]), [-1]+shape_t+[ts[-1]])
