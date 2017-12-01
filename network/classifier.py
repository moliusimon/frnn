from network import Network

import tensorflow as tf


class NetworkClassifier(Network):
    def __init__(self, topology, session=None, scope='model'):
        Network.__init__(self, topology, session=session, scope=scope)

    def _loss(self, placeholders, reduce=True):
        # Make predictions, broadcast targets if required
        y, t = self._apply(placeholders[0]), placeholders[1]
        if len(t.get_shape()) < len(y.get_shape()):
            t = tf.expand_dims(t, axis=0)

        # Binary cross-entropy
        if y.get_shape()[-1].value == 1:
            y = tf.maximum(y, 0.00001)
            loss = - (t * tf.log(y) + (1 - t) * tf.log(1 - y))
            loss = tf.reduce_mean(loss) if reduce else loss

        # Multi-class cross-entropy
        else:
            loss = tf.losses.sparse_softmax_cross_entropy(y, t)

        return loss
