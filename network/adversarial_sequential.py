from adversarial import NetworkAdversarial
import tensorflow as tf
import numpy as np


class NetworkAdversarialSequential(NetworkAdversarial):
    def __init__(self, network_generative, network_discriminative, session=None, scope='model'):
        NetworkAdversarial.__init__(self, network_generative, network_discriminative, session=session, scope=scope)

    def _loss_generative(self, x):
        in_shape = [d.value for d in x.get_shape()]
        full_length, half_length = in_shape[0], in_shape[0] / 2

        # Prepare targets
        td = tf.ones([full_length, tf.shape(x)[1], 1], dtype=np.float32)

        # Apply generative for sequence prediction
        xg = x[:half_length]
        t_yg = self.topology[0]._run(xg, num_ignored=0, num_preds=full_length-half_length)
        yg = tf.concat([xg, t_yg], axis=0)
        loss = self.topology[1]._loss([yg, td], reduce=False)[half_length:]

        # Prepare targets and apply discriminative loss
        return (
            tf.reduce_mean(loss) + tf.reduce_mean((t_yg - x[half_length:]) ** 2),
            tf.reduce_mean(loss),
            tf.reduce_mean(-tf.log(1 - tf.exp(-loss)))
        )

    def _loss_discriminative(self, x, half_batch):
        in_shape = [d.value for d in x.get_shape()]
        full_length, half_length = in_shape[0], in_shape[0] / 2

        # Prepare prediction targets
        t = tf.constant(np.concatenate((
            np.zeros((full_length, half_batch, 1)),
            np.ones((full_length, half_batch, 1))
        ), axis=1), dtype=tf.float32)

        # Pass half the data through the generative network
        xg = x[:half_length, :half_batch]
        yg = self.topology[0]._run(xg, num_ignored=0, num_preds=full_length - half_length)

        # Prepare discriminative input (half from the generative, half from ground truth) and feed discriminator
        xd = tf.concat([tf.concat([xg, yg], axis=0), x[:, half_batch:]], axis=1)
        ret = self.topology[1]._loss([xd, t], reduce=False)[half_length:]

        # Calculate tracking losses
        gen_loss = tf.reduce_mean(-tf.log(1 - tf.exp(-ret[:, :half_batch])))
        dis_loss = tf.reduce_mean(ret[:, :half_batch])

        return tf.reduce_mean(ret), gen_loss, dis_loss
