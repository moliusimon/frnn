import tensorflow as tf
import numpy as np


class Metric:
    def __init__(self):
        pass

    @staticmethod
    def metric_mse(y, t, sample_dims=-1):
        return tf.reduce_mean((y - t) ** 2, axis=sample_dims)

    @staticmethod
    def metric_psnr(y, t, sample_dims=-1, value_range=1):
        return 4.34294481903 * (2 * tf.log(
            tf.cast(value_range, dtype=tf.float32)
        ) - tf.log(
            Metric.metric_mse(y, t, sample_dims=sample_dims)
        ))

    @staticmethod
    def metric_dssim(y, t, sample_dims=None, value_range=1):
        # Set sample dimensions to default value if not provided
        n_dims = len(y.get_shape())
        sample_dims = (n_dims-3, n_dims-2, n_dims-1) if sample_dims is None else sample_dims

        # Check sample dimensions is valid
        if not isinstance(sample_dims, (list, tuple)) or len(sample_dims) != 3:
            raise ValueError('sample_dims must be a list/tuple of length 3 specifying the height,'
                             ' width and channel dimensions of the evaluated data.')

        # Re-order dimensions for processing
        d_order = [i for i in range(n_dims) if i not in sample_dims] + [sample_dims[2], sample_dims[0], sample_dims[1]]
        y, t = tf.transpose(y, d_order), tf.transpose(t, d_order)

        in_shape, out_shape, n_channels = (
            [-1] + [d.value for d in y.get_shape()[-2:]],
            [(-1 if d.value is None else d.value) for d in y.get_shape()[:-2]],
            y.get_shape()[-3].value,
        )

        y, t = tf.reshape(y, in_shape + [1]), tf.reshape(t, in_shape + [1])
        c1, c2 = (value_range * 0.01) ** 2, (value_range * 0.03) ** 2
        gauss = Metric._tf_fspecial_gauss(11, 1.5)

        # Calculate means and squared means
        m_y = tf.nn.conv2d(y, gauss, strides=[1, 1, 1, 1], padding='VALID')
        m_t = tf.nn.conv2d(t, gauss, strides=[1, 1, 1, 1], padding='VALID')
        m2_y, m2_t, m2_yt = m_y ** 2, m_t ** 2, m_y * m_t

        # Calculate variances and covariance
        v_y = tf.nn.conv2d(y * y, gauss, strides=[1, 1, 1, 1], padding='VALID') - m2_y
        v_t = tf.nn.conv2d(t * t, gauss, strides=[1, 1, 1, 1], padding='VALID') - m2_t
        c_yt = tf.nn.conv2d(y * t, gauss, strides=[1, 1, 1, 1], padding='VALID') - m2_yt

        # Calculate SSIM
        ssim = (2 * m2_yt + c1) * (2 * c_yt + c2) / ((m2_t + m2_y + c1) * (v_t + v_y + c2))
        avg_ssim = tf.reduce_mean(tf.reshape(tf.reduce_mean(ssim, axis=[-3, -2, -1]), out_shape), axis=-1)

        # Return DSSIM
        return (1 - avg_ssim) / 2

    @staticmethod
    def metric_crossentropy_bimodal(y, t, sample_dims=-1):
        return -tf.reduce_mean(t * tf.log(y) + (1 - t) * tf.log(1 - y), axis=sample_dims)

    @staticmethod
    def _tf_fspecial_gauss(size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function """
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        g = np.reshape(g / np.sum(g), (size, size, 1, 1))
        return tf.constant(g, dtype=tf.float32)
