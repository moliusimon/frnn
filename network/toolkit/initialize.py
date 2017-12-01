import tensorflow as tf
import numpy as np


def init_weights(shape):
    w = np.random.normal(size=(np.prod(shape[:-1]), shape[-1]))
    u, _, v = np.linalg.svd(w, full_matrices=False)
    w = u if np.prod(shape[:-1]) > shape[-1] else v

    # t_factor = 1 / (np.std(q, axis=0, keepdims=True) * np.sqrt(q.shape[0]))
    t_factor = 2 / (np.std(w, axis=0, keepdims=True) * np.sqrt(w.shape[0] + w.shape[1]))
    return tf.Variable(np.cast[np.float32](t_factor * np.reshape(w, shape)))


def init_biases(shape):
    shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    return tf.Variable(np.zeros(shape, dtype=np.float32))


def init_state_placeholders(placeholders, batch_size):
    """ Initialize network states with zero tensors """
    ret, values = [], {}
    for p, v in placeholders:
        # Gate shape of placeholder and initial value
        p_shape = [d.value for d in p.get_shape()]
        v_shape = [d.value for d in v.get_shape()]
        delta_dims = len(p_shape) - len(v_shape)

        # Reshape value to match dimensions of placeholder
        v = tf.reshape(v, [1] * delta_dims + v_shape)

        # Tile across new dimensions to match shape of placeholder and batch size
        init = values.get(p, tf.tile(v, [batch_size] + p_shape[1:delta_dims] + [1] * len(v_shape)))
        ret.append(init)

    return ret
