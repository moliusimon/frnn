from network.loader import Loader
import numpy as np
import urllib
import cPickle
import gzip
import h5py
import os

# SCRIPT FOR PREPROCESSING THE MMNIST DATASET
#   This script prepares the training and test partitions for the MMNIST dataset. It will take the dataset from the
#   directory specified in DATA_PATH and place the processed partitions in SAVE_PATH. If the dataset is not present, it
#   will automatically download the required files (MNIST in numpy format, and the standard Moving MNIST test
#   partition).

DATA_PATH = '/mnt/DumperA/mmnist/'
SAVE_PATH = '/workmem/'


# ------------------------------------------------------------------
# DEFINE CUSTOM LOADER TO GENERATE TRAINING SAMPLES
# ------------------------------------------------------------------

class LoaderMMNist(Loader):
    def __init__(self, x, sample_length=20):
        self.data = np.reshape(cPickle.load(open(x, 'rb'))[0][0], (-1, 28, 28))
        self.sample_length = sample_length
        self._row = 0
        Loader.__init__(self, len(self.data))

    def _sample(self, indices):
        b_size = len(indices)

        start_y, start_x = self._get_random_trajectory(b_size * 2)
        data = np.zeros((b_size, self.sample_length, 64, 64), dtype=np.float32)
        for j in xrange(b_size):
            for n in xrange(2):
                ind = self._row
                self._row += 1
                if self._row == self.num_samples:
                    self._row = 0
                    np.random.shuffle(self.data)
                digit_image = self.data[ind, :, :]
                for i in xrange(self.sample_length):
                    top, left = start_y[i, j * 2 + n], start_x[i, j * 2 + n]
                    bottom, right = top + 28, left + 28
                    data[j, i, top:bottom, left:right] = np.maximum(data[j, i, top:bottom, left:right], digit_image)

        return np.cast[np.uint8](255 * np.expand_dims(data, 4))

    def _get_random_trajectory(self, batch_size):
        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((self.sample_length, batch_size))
        start_x = np.zeros((self.sample_length, batch_size))
        for i in xrange(self.sample_length):
            # Take a step along velocity.
            x, y = x + v_x * 0.1, y + v_y * 0.1

            # Bounce off edges.
            for j in xrange(batch_size):
                if x[j] <= 0:
                    x[j], v_x[j] = 0, -v_x[j]
                if x[j] >= 1.0:
                    x[j], v_x[j] = 1.0, -v_x[j]
                if y[j] <= 0:
                    y[j], v_y[j] = 0, -v_y[j]
                if y[j] >= 1.0:
                    y[j], v_y[j] = 1.0, -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = np.cast[np.int32]((64 - 28) * start_y)
        start_x = np.cast[np.int32]((64 - 28) * start_x)
        return start_y, start_x


def prepare_training(path_in, path_out, n_samples):
    n_samples = n_samples + n_samples % 32
    n_steps = n_samples / 32

    # Prepare destination file
    fp = h5py.File(path_out, 'w')

    # Start sampling data
    loader = LoaderMMNist(path_in)
    for i in range(n_steps):
        videos = loader.sample(mode='train', batch_size=32)[0]
        for j, v in enumerate(videos):
            ds = fp.create_dataset(str(32*i + j), (20, 64, 64, 1), dtype='u1')
            ds[...] = v

    # Close HDF5 file
    fp.close()


def prepare_testing(path_in, path_out):
    # Prepare input/output files
    fi = np.cast[np.uint8](255 * np.load(path_in))
    fo = h5py.File(path_out, 'w')

    # Create a dataset for each video in input file
    for i, v in enumerate(fi):
        ds = fo.create_dataset(str(i), (20, 64, 64, 1), dtype='u1')
        ds[...] = v

    # Close HDF5 file
    fo.close()


if __name__ == '__main__':
    # Download MNIST if not present
    if not os.path.exists(DATA_PATH + 'mnist.pkl'):
        print "Downloading MNIST dataset ..."
        urllib.URLopener().retrieve(
            'http://deeplearning.net/data/mnist/mnist.pkl.gz',
            DATA_PATH + 'mnist.pkl.gz'
        )

        print 'Extracting MNIST dataset ...'
        with gzip.open(DATA_PATH + 'mnist.pkl.gz', 'rb') as f_in, open(DATA_PATH + 'mnist.pkl', 'wb') as f_out:
            f_out.write(f_in.read())

        print 'Removing compressed MNIST version ...'
        os.remove(DATA_PATH + 'mnist.pkl.gz')

    # Download Moving MMNIST test data if not present
    if not os.path.exists(DATA_PATH + 'mmnist_test.npy'):
        print 'Downloading Moving MNIST test data ...'
        urllib.URLopener().retrieve(
            'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy',
            DATA_PATH + 'mmnist_test.npy'
        )

    # Prepare train partition
    print 'Generating train data ...'
    prepare_training(
        DATA_PATH + 'mnist.pkl',
        SAVE_PATH + 'mmnist_train.hdf5',
        n_samples=1000000
    )

    # Prepare test partition
    print 'Preparing test data ...'
    prepare_testing(
        DATA_PATH + 'mmnist_test.npy',
        SAVE_PATH + 'mmnist_test.hdf5'
    )