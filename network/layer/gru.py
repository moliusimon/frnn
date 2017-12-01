from layer import Layer

import tensorflow as tf


class LayerGru(Layer):
    def __init__(self, shape):
        self.shape = shape
        self.input_shape = shape[0]
        Layer.__init__(self, True, [
            {'name': 'b1',           'type': 'b', 'shape': shape[-1]},
            {'name': 'b2',           'type': 'b', 'shape': shape[-1]},
            {'name': 'b3',           'type': 'b', 'shape': shape[-1]},
            {'name': 'w1',           'type': 'w', 'shape': (shape[0]+shape[-1], shape[-1])},
            {'name': 'w2',           'type': 'w', 'shape': (shape[0]+shape[-1], shape[-1])},
            {'name': 'w3',           'type': 'w', 'shape': (shape[0]+shape[-1], shape[-1])},
            {'name': '_state_init',  'type': 'b', 'shape': (shape[-1],)}
        ])

        # Define state placeholder
        self.state = tf.placeholder(tf.float32, [None, self.shape[-1]])
        self._state = self.state

    def forward(self, x):
        # Calculate output of first two gates (sigmoid)
        x1 = tf.concat([x, self._state], axis=1)
        g1 = tf.nn.sigmoid(tf.matmul(x1, self.w1) + self.b1)
        g2 = tf.nn.sigmoid(tf.matmul(x1, self.w2) + self.b2)

        # Calculate output of third gate (tanh)
        x2 = tf.concat([x, self._state*g1], axis=1)
        g3 = tf.nn.tanh(tf.matmul(x2, self.w3) + self.b3)

        self._state = g2 * g3 + (1-g2) * self._state
        return self._state

    def _gather(self, placeholder=False):
        return [(self.state, self._state_init) if placeholder else self._state]

    def _scatter(self, values):
        self._state = values.pop(0)

    def get_input_shape(self):
        return [self.shape[0]]

    def get_output_shape(self):
        return [self.shape[-1]]
