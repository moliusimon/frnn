import tensorflow as tf
import numpy as np


class Memory:
    def __init__(self, shape):
        self.shape = shape
        self.content_dimensions = len(self.shape) - 1

        # Define memory initial & placeholder states
        self.memory = tf.placeholder(tf.float32, (None,) + self.shape)
        self._memory_init = tf.constant(np.zeros(self.shape, dtype=np.float32))
        self._memory = self.memory

    def get_num_cells(self):
        return self.shape[0]

    def get_cell_size(self):
        return np.prod(self.shape[1:])

    def get_cell_shape(self):
        return self.shape[1:]

    def read_cells(self, cell_weights):
        weights = self._prepare_indexing_weights(cell_weights / tf.reduce_sum(cell_weights, keep_dims=True))
        return tf.reduce_sum(weights * self._memory, axis=1)

    def write_cells(self, cell_weights, values):
        weights = self._prepare_indexing_weights(cell_weights)
        values = self._prepare_cell_contents(values)
        self._memory = self._memory + weights * values

    def erase_cells(self, cell_weights, signal_is_keep=False):
        weights = self._prepare_indexing_weights(cell_weights)
        self._memory = self._memory * (weights if signal_is_keep else (1 - weights))

    # ---------------------------------------------------------------------------------
    # -- PARAMETER GATHERING METHODS
    # ---------------------------------------------------------------------------------

    def gather(self, placeholder=False):
        ret = [(self.memory, self._memory_init) if placeholder else self._memory]

        # Gather from child class if function defined
        if hasattr(self, '_gather'):
            t_ret = getattr(self, '_gather')(placeholder)
            ret = ret + (t_ret if isinstance(t_ret, list) else [t_ret])

        return ret

    def scatter(self, values):
        self._memory = values.pop(0)
        if hasattr(self, '_scatter'):
            getattr(self, '_scatter')(values)

    # ---------------------------------------------------------------------------------
    # -- PRIVATE CLASS METHODS
    # ---------------------------------------------------------------------------------

    def _prepare_indexing_weights(self, weights):
        # Reshape weights
        new_dims = self.content_dimensions - (len(weights.get_shape()) - 1)
        return tf.reshape(weights, tf.concat((tf.shape(weights), [1] * new_dims), axis=0))

    def _prepare_cell_contents(self, values):
        # Reshape contents
        new_dims = self.content_dimensions - (len(values.get_shape()) - 1)
        return tf.reshape(values, tf.concat(([tf.shape(values)[0]], [1] * new_dims, tf.shape(values)[1:]), axis=0))
