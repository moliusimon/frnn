from network import Network
import tensorflow as tf


class NetworkRegressor(Network):
    def __init__(self, topology, session=None, scope='model'):
        Network.__init__(self, topology, session=session, scope=scope)

    def _loss(self, placeholders):
        y, t = self._apply(placeholders[0]), placeholders[1]
        if len(t.get_shape()) < len(y.get_shape()):
            t = tf.expand_dims(t, axis=0)

        return tf.reduce_mean((y - t) ** 2)
