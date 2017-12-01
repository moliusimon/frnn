from network import Network
from toolkit import init_state_placeholders

import tensorflow as tf


class NetworkAutoencoder(Network):
    def __init__(self, topology, session=None, scope='model'):
        Network.__init__(self, topology, session=session, scope=scope)

    def _loss(self, placeholders):
        y, t = self._apply(placeholders[0]), placeholders[0]
        if self.topology[0].is_recurrent():
            y, t = y[:-1], t[1:]

        return tf.reduce_mean((t - y) ** 2)

    def _apply(self, x):
        b_size = tf.shape(x)[1]

        # If regular topology, do forward pass
        if not self.topology[0].is_recurrent():
            y = self.topology[0].forward(x)
            return self.topology[0].backward(y)

        # If recurrent topology, initialize and cycle
        stub_output = tf.zeros([b_size] + self.topology[0].get_input_shape(), dtype=tf.float32)
        state_inits = init_state_placeholders(self.topology[0].gather(placeholder=True), batch_size=b_size)
        state_final = tf.scan(self._cycle, x, initializer=[stub_output] + state_inits)
        return state_final[0]

    def _cycle(self, state, x):
        self.topology[0].scatter(state[1:])
        y = self.topology[0].forward(x)
        y = self.topology[0].backward(y)
        return [y] + self.topology[0].gather()
