from network.preprocessor import Preprocessor
from network import NetworkFolded
from network.topology import TopologyFolded
from loader import LoaderMMNIST
import tensorflow as tf

# ------------------------------------------------------------------
# CONFIGURE MODEL
# ------------------------------------------------------------------

DATA_ROOT = '/workmem/'
DATA_TRAIN = DATA_ROOT + 'mmnist_train.hdf5'
DATA_TEST = DATA_ROOT + 'mmnist_test.hdf5'
SAVE_ROOT = './save/mmnist_frnn/'
SAVE_PATH = SAVE_ROOT + 'model/model.ckpt'
TRAIN_STEPS = 500000
BATCH_SIZE = 12
DEVICE = '/gpu:0'

MODEL = NetworkFolded(topology=TopologyFolded([
    {'type': 'convolutional', 'input_shape': (64, 64, 1),    'shape': (5, 5, 1, 32),     'activation': tf.nn.tanh},
    {'type': 'convolutional', 'input_shape': (64, 64, 32),   'shape': (5, 5, 32, 64),    'activation': tf.nn.tanh},
    {'type': 'pooling',       'pool_size': (2, 2)},
    {'type': 'bconvgru',      'input_shape': (32, 32, 64),   'shape': (5, 5, 64, 128)},
    {'type': 'bconvgru',      'input_shape': (32, 32, 128),  'shape': (5, 5, 128, 128)},
    {'type': 'pooling',       'pool_size': (2, 2)},
    {'type': 'bconvgru',      'input_shape': (16, 16, 128),  'shape': (5, 5, 128, 256)},
    {'type': 'bconvgru',      'input_shape': (16, 16, 256),  'shape': (5, 5, 256, 256)},
    {'type': 'pooling',       'pool_size': (2, 2)},
    {'type': 'bconvgru',      'input_shape': (8, 8, 256),    'shape': (3, 3, 256, 512)},
    {'type': 'bconvgru',      'input_shape': (8, 8, 512),    'shape': (3, 3, 512, 512)},
    {'type': 'pooling',       'pool_size': (2, 2)},
    {'type': 'bconvgru',      'input_shape': (4, 4, 512),    'shape': (3, 3, 512, 256)},
    {'type': 'bconvgru',      'input_shape': (4, 4, 256),    'shape': (3, 3, 256, 256)},
]), scope='flownet', loss='l1')

# Prepare train data loader and preprocessor
TRAIN_PREPROCESSOR = Preprocessor([
    {'type': 'swapaxes', 'order': [1, 0, 2, 3, 4]},
    {'type': 'rescale', 'weight': 2, 'bias': -1},
], loader=LoaderMMNIST(DATA_TRAIN))

# Prepare test data loader and preprocessor
TEST_PREPROCESSOR = Preprocessor([
    {'type': 'swapaxes', 'order': [1, 0, 2, 3, 4]},
    {'type': 'rescale', 'weight': 2, 'bias': -1}
], loader=LoaderMMNIST(DATA_TEST))
