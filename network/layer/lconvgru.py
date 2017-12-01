from layer import Layer

import tensorflow as tf
import numpy as np


class LayerLConvgru(Layer):
    def __init__(self, shape, input_shape):
        self.shape = shape
        self.input_shape = input_shape

        fgate_shape = shape[:-2] + (2*shape[-2] + shape[-1], shape[-1])
        bgate_shape = shape[:-2] + (shape[-2] + 2*shape[-1], shape[-2])
        Layer.__init__(self, True, [
            {'name': 'b1',               'type': 'b', 'shape': shape[-1]},
            {'name': 'b2',               'type': 'b', 'shape': shape[-1]},
            {'name': 'b3',               'type': 'b', 'shape': shape[-1]},
            {'name': 'w1',               'type': 'w', 'shape': fgate_shape},
            {'name': 'w2',               'type': 'w', 'shape': fgate_shape},
            {'name': 'w3',               'type': 'w', 'shape': fgate_shape},
            {'name': '_state_init',      'type': 'b', 'shape': (shape[-1],)},
            {'name': 'b1_back',          'type': 'b', 'shape': shape[-2]},
            {'name': 'b2_back',          'type': 'b', 'shape': shape[-2]},
            {'name': 'b3_back',          'type': 'b', 'shape': shape[-2]},
            {'name': 'w1_back',          'type': 'w', 'shape': bgate_shape},
            {'name': 'w2_back',          'type': 'w', 'shape': bgate_shape},
            {'name': 'w3_back',          'type': 'w', 'shape': bgate_shape},
            {'name': '_state_init_back', 'type': 'b', 'shape': (shape[-2],)}
        ])

        # Define state placeholder
        self.state = tf.placeholder(tf.float32, [None] + list(self.input_shape[:-1]) + [self.shape[-1]])
        self._state = self.state

        # Define backwards state placeholder
        self.state_back = tf.placeholder(tf.float32, [None] + list(self.input_shape))
        self._state_back = self.state_back

    def forward(self, x):
        x = [x, self._state_back]

        # Calculate output of first two gates (sigmoid)
        x1 = tf.concat(x + [self._state], axis=-1)
        g1 = tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(x1, self.w1, [1, 1, 1, 1], 'SAME'), self.b1))
        g2 = tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(x1, self.w2, [1, 1, 1, 1], 'SAME'), self.b2))

        # Calculate output of third gate (tanh)
        x2 = tf.concat(x + [self._state*g1], axis=-1)
        g3 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(x2, self.w3, [1, 1, 1, 1], 'SAME'), self.b3))

        self._state = g2 * g3 + (1-g2) * self._state
        return self._state

    def backward(self, x):
        x = [x, self._state]

        # Calculate output of first two gates (sigmoid)
        x1 = tf.concat(x + [self._state_back], axis=-1)
        g1 = tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(x1, self.w1_back, [1, 1, 1, 1], 'SAME'), self.b1_back))
        g2 = tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(x1, self.w2_back, [1, 1, 1, 1], 'SAME'), self.b2_back))

        # Calculate output of third gate (tanh)
        x2 = tf.concat(x + [self._state_back*g1], axis=-1)
        g3 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(x2, self.w3_back, [1, 1, 1, 1], 'SAME'), self.b3_back))

        self._state_back = g2 * g3 + (1-g2) * self._state_back
        return self._state_back

    def _gather(self, placeholder=False):
        return [
            (self.state, self._state_init), (self.state_back, self._state_init_back)
        ] if placeholder else [self._state, self._state_back]

    def _scatter(self, values):
        self._state = values.pop(0)
        self._state_back = values.pop(0)

    def get_input_shape(self):
        return list(self.input_shape)

    def get_output_shape(self):
        return list(self.input_shape[:-1]) + [self.shape[-1]]
