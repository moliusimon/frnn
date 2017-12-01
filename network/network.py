from preprocessor import Preprocessor
from topology import Topology
from toolkit import init_state_placeholders
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob
import os


class Network:
    def __init__(self, topology, session, scope):
        # Create session if not provided, store variable scope name
        self.session = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True)
        ) if session is None else session
        self.scope = scope

        # Build network topologies
        with tf.variable_scope(self.scope):
            self.topology = topology if isinstance(topology, (list, tuple)) else [topology]
            to_build = [t for t in self.topology if isinstance(t, Topology)]
            for i_t, t in enumerate(to_build):
                sub_scope = 'topology_' + str(i_t)
                with tf.variable_scope(sub_scope):
                    t.build(self.scope + '/' + sub_scope)

        # Initialize network parameters
        self.session.run(tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        ))

        # Prepare step counter & saver
        self.step = 0
        self.saver = tf.train.Saver(
            self.get_model_variables(),
            max_to_keep=5,
        )

    # DEFINE MODEL SAVING/RESTORATION METHODS
    # -------------------------------------------------------------------

    def save(self, save_path):
        # Create save path if it doesn't exist
        directory = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save model
        self.saver.save(self.session, save_path, global_step=self.step)

    def load(self, save_path, step=None):
        # Use specified save step, otherwise find latest
        step = np.max([
            int(f.split('-')[-1].split('.')[0]) for f in glob.glob(save_path + '-*.index')
        ]) if step is None else step

        # Load saved model
        self.saver.restore(self.session, save_path + '-' + str(step))
        self.step = step

    def get_model_variables(self, component=None):
        components = self.topology if component is None else [component]
        return [v for c in components for v in c.get_model_variables()]

    # DEFINE TRAIN/TEST METHODS
    # -------------------------------------------------------------------

    def train(
        self,
        x, val=None, val_frequency=100,
        batch_size=32, iterations=100000, optimizer=None,
        save_path=None, save_frequency=500
    ):
        # Prepare data loaders, placeholders, function feeds & outputs
        loader = x if isinstance(x, Preprocessor) else Preprocessor([], loader=x)
        loader_val = val if val is None else (val if isinstance(val, Preprocessor) else Preprocessor([], loader=val))
        placeholders = loader.get_placeholders()
        loss, accuracy, feed = self._untangle_function_rets(self._loss(*placeholders))
        accuracy = loss if accuracy is None else accuracy

        # Build optimizer
        optimizer_scope = self.scope + '/optimizer'
        with tf.variable_scope(optimizer_scope):
            optimizer = tf.train.RMSPropOptimizer(0.0001) if optimizer is None else optimizer
            fcn = optimizer.minimize(
                tf.reduce_mean(loss) if isinstance(loss, list) else loss,
                var_list=self.get_model_variables()
            )

        # Initialize optimizer variables
        self.session.run(tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=optimizer_scope)
        ))

        # Start optimization
        runs = [fcn] + (loss if isinstance(loss, list) else [loss]) + [accuracy]
        for self.step in range(self.step+1, self.step+1+iterations):
            batch_data = loader.sample(batch_size, 'train', full_augment=False)
            feed.update({k: v for k, v in zip(placeholders, batch_data)})
            i_loss, i_accuracy = self.session.run(runs, feed_dict=feed)[1:]

            # Print loss and save/validate if required
            print 'Iteration ' + str(self.step) + ' loss: ' + str(i_loss) + ' error: ' + str(i_accuracy)
            if self.step % save_frequency == 0:
                if save_path is not None:
                    self.save(save_path)

                if val is not None:
                    val_accruacy = []
                    for i in range(10):
                        batch_data = loader_val.sample(batch_size, 'train', full_augment=False)
                        feed.update({k: v for k, v in zip(placeholders, batch_data)})
                        val_accruacy.append(self.session.run(accuracy, feed_dict=feed))

                    print 'Validation accuracy: ' + str(np.mean(val_accruacy))

        # Save model if path provided
        if save_path is not None:
            self.save(save_path)

    def test(self, x, batch_size=32, metric='mse'):
        # Prepare data loader, placeholders, function feeds & outputs
        loader = x if isinstance(x, Preprocessor) else Preprocessor([], loader=x)
        placeholders = loader.get_placeholders()

        error, baseline, feed = self._untangle_function_rets(self._test(*placeholders, metric=metric))
        to_run = [error, baseline] if baseline is not None else error

        # Calculate error for all samples, one batch at a time
        errors, baselines = [], []
        while loader.has_batch():
            batch_data = loader.sample(batch_size, 'test', full_augment=True)
            feed.update({k: v for k, v in zip(placeholders, batch_data)})

            # Evaluate network on batch
            t_rets = self.session.run(to_run, feed_dict=feed)
            t_errors, t_baselines = t_rets if isinstance(t_rets, (tuple, list)) else (t_rets, None)

            # Store test results
            if t_errors is not None:
                errors.append(t_errors)
            if t_baselines is not None:
                baselines.append(t_baselines)

        # Concatenate batch results and return
        return np.concatenate(errors, axis=0), np.concatenate(baselines, axis=0)

    def run(self, x, batch_size=32, plot=False):
        # Prepare data loader, placeholders, function feeds & outputs
        loader = x if isinstance(x, Preprocessor) else Preprocessor([], loader=x)
        placeholders = loader.get_placeholders()
        predictions, _, feed = self._untangle_function_rets(self._run(*placeholders))

        # Generate predictions for all samples, one batch at a time
        ret = []
        while loader.has_batch():
            batch_data = loader.sample(batch_size, 'valid', full_augment=True)
            feed.update({k: v for k, v in zip(placeholders, batch_data)})
            t_predictions = np.transpose(self.session.run(predictions, feed_dict=feed), axes=[1, 0, 2, 3, 4])
            ret.append(t_predictions)

            # If asked to, plot results
            if plot:
                # Prepare colormap
                cmap = None if t_predictions.shape[-1] == 3 else 'gray'
                t_groundtruth = batch_data[0][10:]
                if cmap is 'gray':
                    t_predictions = t_predictions[..., 0]
                    t_groundtruth = t_groundtruth[..., 0]

                # Plot images (grayscale)
                for i in range(len(t_predictions)):
                    for j in range(0, 10):
                        plt.subplot(4, 5, 1+j)
                        plt.imshow(t_predictions[i, j] / 2 + 0.5, cmap=cmap)
                        plt.subplot(4, 5, 11+j)
                        plt.imshow(t_groundtruth[j, i] / 2 + 0.5, cmap=cmap)
                    plt.show()

        return np.concatenate(tuple(ret), axis=0)

    def _loss(self, placeholders):
        raise NotImplementedError('Loss function not implemented for this network type!')

    def _test(self, placeholders, metric='mse'):
        raise NotImplementedError('Test function not implemented for this network type!')

    def _run(self, placeholders):
        raise NotImplementedError('Run function not implemented for this network type!')

    # HELPER METHODS
    # -------------------------------------------------------------------

    def _apply(self, x):
        b_size = tf.shape(x)[1]

        # If regular topology, do forward pass
        if not self.topology[0].is_recurrent():
            return self.topology[0].forward(x)

        # If recurrent topology, initialize and cycle
        stub_output = tf.zeros([b_size] + self.topology[0].get_output_shape(), dtype=tf.float32)
        state_inits = init_state_placeholders(self.topology[0].gather(placeholder=True), batch_size=b_size)
        state_final = tf.scan(self._cycle, x, initializer=[stub_output] + state_inits)
        return state_final[0]

    def _cycle(self, state, x):
        self.topology[0].scatter(state[1:])
        y = self.topology[0].forward(x)
        return [y] + self.topology[0].gather()

    @staticmethod
    def _untangle_function_rets(rets):
        # If not iterable, return self as result, without metric and with a stub feed dictionary
        if not isinstance(rets,  (tuple, list)):
            return rets, None, {}

        # Capture function metric and feed dictionary if present in the return
        result, metric, feeds = rets[0], None, {}
        for v in rets[1:]:
            # If a tensor, consider as baseline/accuracy
            if isinstance(v, tf.Tensor):
                metric = v

            # If a dictionary, consider as feed dictionary
            if isinstance(v, tuple):
                feeds = v

        return result, metric, feeds