from network import Network
from toolkit import init_state_placeholders
from toolkit.metric import Metric
import tensorflow as tf


class NetworkFolded(Network):
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

        deep_init = [tf.zeros(
            (b_size,) + l.input_shape
        ) for l in self.topology[0].topology if hasattr(l, 'input')]

        state_inits = init_state_placeholders(self.topology[0].gather(placeholder=True), batch_size=b_size)
        state_feeds = tf.scan(self._cycle_feed, x, initializer=[deep_init, state_inits])
        state_preds = tf.scan(self._cycle_predict, [tf.range(0, limit=num_preds, dtype=tf.float32)], initializer=[
            x[0], [s[-1] for s in state_feeds[0]], [s[-1] for s in state_feeds[1]]
        ]) if num_preds > 0 else [None, None, None]

        # Return output and states from encoding and decoding
        return state_preds[0], state_feeds[0], state_preds[1]

    def _cycle_feed(self, state, inputs):
        # Scatter states, pipe through network
        self.topology[0].scatter(state[1])
        y, y_down = self.topology[0].forward(inputs)

        # Gather and return states
        return [y_down] + [self.topology[0].gather(placeholder=False)]

    def _cycle_predict(self, state, inputs):
        # Scatter states, set layer inputs (backwards state)
        self.topology[0].scatter(state[2])
        convgrus = [l for l in self.topology[0].topology if hasattr(l, 'input')]
        for l, sin in zip(convgrus, state[1]):
            l.input = sin

        # Pass through network (generate current frame, then update state)
        y, y_up = self.topology[0].backward(self.topology[0].topology[-1].gather()[0])
        self.topology[0].update_state()

        # Gather and return states
        return [y, y_up] + [self.topology[0].gather(placeholder=False)]

