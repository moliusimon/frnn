from network import Network
from toolkit import init_state_placeholders
from toolkit.metric import Metric
import tensorflow as tf


class NetworkRLadder(Network):
    def __init__(self, topology, session=None, scope='model', loss='l2'):
        self.loss_type = loss
        Network.__init__(self, [topology], session=session, scope=scope)

    def _loss(self, x, num_ignored=10):
        # Get predictions & targets
        y = self._apply(x[:-num_ignored], num_preds=num_ignored)[0]
        t = x[-num_ignored:]

        # Calculate loss
        loss = {
            'l1': self._loss_l1,
            'l2': self._loss_l2,
        }[self.loss_type](y, t)

        # Calculate error
        error = tf.reduce_mean((y - t) ** 2)

        return loss, error

    @staticmethod
    def _loss_l1(y, t):
        return tf.reduce_mean(tf.abs(y - t))

    @staticmethod
    def _loss_l2(y, t):
        return tf.reduce_mean((y - t) ** 2)

    def _test(self, x, num_ignored=10, metric='mse'):
        # Make predictions, prepare targets and baseline
        y = (self._apply(x[:-num_ignored], num_preds=num_ignored)[0] + 1) / 2
        t, b = (x[-num_ignored:] + 1) / 2, (tf.stack([x[-num_ignored-1]] * num_ignored, axis=0) + 1) / 2

        # Calculate error metrics
        ret = ([], [])
        for m in (metric if isinstance(metric, (tuple, list)) else (metric,)):
            # Calculate specific metric for predictions and baseline
            err_f = getattr(Metric, 'metric_' + m)
            e_pred = tf.transpose(err_f(y, t, sample_dims=[2, 3, 4]), perm=(1, 0))
            e_bline = tf.transpose(err_f(b, t, sample_dims=[2, 3, 4]), perm=(1, 0))

            # Append metrics to return tensors
            ret[0].append(e_pred)
            ret[1].append(e_bline)

        # Stack error metrics and return
        return (
            tf.stack(ret[0], axis=1),
            tf.stack(ret[1], axis=1),
        )

    def _run(self, x, num_ignored=10):
        return self._apply(x[:-num_ignored], num_preds=num_ignored)[0]

    # HELPER (NON-PROTOTYPE) METHODS
    # --------------------------------------------------------

    def _apply(self, x, num_preds=1):
        b_size = tf.shape(x)[1]

        # Prepare initial state values, stub inputs for prediction
        state_inits = init_state_placeholders(self.topology[0].gather(placeholder=True), batch_size=b_size)
        z = tf.zeros((num_preds, 1), dtype=tf.float32)

        # Feed inputs, predict outputs
        state_feeds = tf.scan(self._cycle_feed, x[:-1], initializer=state_inits)
        state_preds = tf.scan(
            self._cycle_predict, z, initializer=[x[-1], [s[-1] for s in state_feeds]]
        ) if num_preds > 0 else [None, None]

        # Return output and states from encoding and decoding
        return state_preds[0], state_feeds, state_preds[1]

    def _cycle_feed(self, state, inputs):
        # Scatter states, pipe through network
        self.topology[0].scatter(state)
        y = self.topology[0].forward(inputs)
        y = self.topology[0].backward(y)

        # Gather and return states
        return self.topology[0].gather(placeholder=False)

    def _cycle_predict(self, state, inputs):
        # Scatter states, pipe through network
        self.topology[0].scatter(state[1])
        y = self.topology[0].forward(state[0])
        y = self.topology[0].backward(y)

        # Gather and return output, states
        return [y, self.topology[0].gather(placeholder=False)]
