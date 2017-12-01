from network import Network
from preprocessor import Preprocessor
import tensorflow as tf
import numpy as np


class NetworkAdversarial(Network):
    def __init__(self, network_generative, network_discriminative, session=None, scope='model'):
        Network.__init__(self, [network_generative, network_discriminative], session=session, scope=scope)

    def train(self, x, batch_size=32, iterations=100000, optimizer=None, save_path=None, save_frequency=500):
        # Prepare data loading and pre-processing
        loader = x if isinstance(x, Preprocessor) else Preprocessor([], loader=x)

        # Prepare placeholders and target constant
        placeholder = loader.get_placeholders()[0]

        # Prepare losses
        loss_generative, g_track_gen, g_track_dis = self._loss_generative(placeholder)
        loss_discriminative, d_track_gen, d_track_dis = self._loss_discriminative(placeholder, batch_size/2)

        # Build optimizers for network
        opt_scope = self.scope + '/optimizer'
        with tf.variable_scope(opt_scope):
            optimizer_g = tf.train.RMSPropOptimizer(0.0001) if optimizer is None else optimizer
            optimizer_d = tf.train.GradientDescentOptimizer(0.01)

            # Build generative optimizer
            fcn_gen = optimizer_g.minimize(
                loss_generative,
                var_list=self.topology[0].get_model_variables()
            )

            # Build discriminative optimizer
            fcn_dis = optimizer_d.minimize(
                loss_discriminative,
                var_list=self.topology[1].get_model_variables()
            )

        # Initialize optimizer variables
        self.session.run(tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope)
        ))

        g_loss, d_loss = 0., 0.
        runs_generative = [fcn_gen, g_track_gen, g_track_dis]
        runs_discriminative = [fcn_dis, d_track_gen, d_track_dis]
        for i in range(1, iterations+1):
            # Train discriminative network
            if g_loss <= d_loss:
                batch_data = loader.sample(batch_size, 'train', full_augment=False)[0]
                g_loss, d_loss = self.session.run(runs_discriminative, feed_dict={placeholder: batch_data})[1:]
                print 'Iteration ' + str(i) + '(trained disc.) losses: ' + str(g_loss) + 'g ' + str(d_loss) + 'd'

            # Train generative network
            if d_loss <= g_loss:
                batch_data = loader.sample(batch_size, 'train', full_augment=False)[0]
                g_loss, d_loss = self.session.run(runs_generative, feed_dict={placeholder: batch_data})[1:]
                print 'Iteration ' + str(i) + '(trained gene.) losses: ' + str(g_loss) + 'g ' + str(d_loss) + 'd'

            # Save model if required
            if save_path is not None and i % save_frequency == 0:
                self.save(save_path)

        # Save model if path provided
        if save_path is not None:
            self.save(save_path)

    def _loss_generative(self, x):
        # Prepare targets
        td = tf.ones([tf.shape(x)[0], 1], dtype=np.float32)

        # Apply generative
        yg = self.topology[0]._run(x)
        loss = self.topology[1]._loss([yg, td], reduce=False)

        # Prepare targets and apply discriminative loss
        return tf.reduce_mean(loss), tf.reduce_mean(loss), tf.reduce_mean(-tf.log(1 - tf.exp(-loss)))

    def _loss_discriminative(self, x, half_batch):
        # Prepare prediction targets
        t = tf.constant(np.concatenate((
            np.zeros((half_batch, 1)),
            np.ones((half_batch, 1))
        ), axis=0), dtype=tf.float32)

        # Prepare discriminative input (pass half the instances through the generative)
        xg = x[:half_batch]
        yg = self.topology[0]._run(xg)
        xd = tf.concat([yg, x[half_batch:]], axis=0)

        # Apply loss
        ret = self.topology[1]._loss([xd, t], reduce=False)
        gen_loss = tf.reduce_mean(-tf.log(1 - tf.exp(-ret[:half_batch])))
        dis_loss = tf.reduce_mean(ret[:half_batch])

        return tf.reduce_mean(ret), gen_loss, dis_loss
