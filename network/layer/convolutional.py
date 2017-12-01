from layer import Layer

import tensorflow as tf


class LayerConvolutional(Layer):
    def __init__(self, shape, input_shape, activation=tf.nn.tanh):
        self.shape = shape
        self.input_shape = input_shape
        self.activation = (lambda x: x) if activation is None else activation
        Layer.__init__(self, True, [
            {'name': 'b1', 'type': 'b', 'shape': self.shape[-1]},
            {'name': 'b2', 'type': 'b', 'shape': self.shape[-2]},
            {'name': 'w',  'type': 'w', 'shape': self.shape}
        ])

    def forward(self, x):
        x = tf.concat(x, -1) if isinstance(x, (list, tuple)) else x
        dims = len(self.shape)

        # 2D convolution
        if dims == 4:
            conv_res = tf.nn.conv2d(x, self.w, [1] * dims, padding='SAME')

        # 3D convolution
        if dims == 5:
            conv_res = tf.nn.conv3d(x, self.w, [1] * dims, padding='SAME')

        return self.activation(tf.nn.bias_add(conv_res, self.b1))

    def backward(self, x):
        dims = len(self.shape)

        # 2D convolution
        if dims == 4:
            w = tf.reverse(tf.transpose(self.w, perm=[0, 1, 3, 2]), [0, 1])
            conv_res = tf.nn.conv2d(x, w, [1] * dims, padding='SAME')

        # 3D convolution
        if dims == 5:
            w = tf.reverse(tf.transpose(self.w, perm=[0, 1, 2, 4, 3]), [0, 1, 2])
            conv_res = tf.nn.conv3d(x, w, [1] * dims, padding='SAME')

        return self.activation(tf.nn.bias_add(conv_res, self.b2))

    def get_input_shape(self):
        return list(self.input_shape)

    def get_output_shape(self):
        return list(self.input_shape[:-1]) + [self.shape[-1]]
