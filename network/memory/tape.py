from memory import Memory
import tensorflow as tf
import numpy as np


class MemoryTape(Memory):
    def __init__(self, shape):
        Memory.__init__(self, shape)

        # Prepare pointer state
        self.pointer = tf.placeholder(tf.float32, (None, self.shape[0]))
        self._pointer_init = tf.constant(np.zeros((self.shape[0],), dtype=tf.float32))
        self._pointer = self.pointer

    def write(self, weights, values):
        """
        Write 'values' at pointer location with intensity equal to 'weights'
        :param weights: Write signal strength (BATCH_SIZE[x...])
        :param values: Values to write (BATCH_SIZE[x...])
        """

        self.write_cells(self._prepare_weights(weights), values)

    def erase(self, weights):
        """
        Erase at pointer location with intensity equal to 'weights'.
        :param weights: Erase signal strength (BATCH_SIZE[x...])
        """

        self.erase_cells(self._prepare_weights(weights))

    def move_pointer(self, delta):
        pass  # TODO

    def _prepare_weights(self, weights):
        weights = tf.expand_dims(weights, axis=1)
        extra_dims = len(weights.get_shape()) - len(self._pointer.get_shape())
        pointer_weights = tf.reshape(self._pointer, tf.concat((tf.shape(self._pointer), [1] * extra_dims), axis=0))
        return pointer_weights * weights

    # ---------------------------------------------------------------------------------
    # -- PARAMETER GATHERING METHODS
    # ---------------------------------------------------------------------------------

    def _gather(self, placeholder=False):
        return [(self.pointer, self._pointer_init) if placeholder else self._pointer]

    def _scatter(self, values):
        self._pointer = values.pop(0)
