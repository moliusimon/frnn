import imageio
import glob
import scipy.ndimage as ndim
import scipy.misc as sm
import numpy as np


def preprocess_predictions(frames, height=66, width=82):
    frames = [(f if len(f.shape) == 3 else np.expand_dims(f, axis=2)) for f in frames]
    frames = [(f if f.shape[2] == 3 else np.concatenate([f, f, f], axis=2)) for f in frames]
    frames = frames[:5] + [sm.imresize(f, frames[0].shape[:2]) for f in frames[5:]]

    # Pad frames to fit expected width
    padding = (((height - frames[0].shape[0]) / 2,)*2, ((width - frames[0].shape[1]) / 2,)*2, (0, 0))
    frames = [np.pad(f, padding, 'constant', constant_values=255) for f in frames]

    # Add border to predictions
    for i in range(5, len(frames)):
        frames[i][:1, :, ...] = [[[255, 0, 0]]]
        frames[i][-1:, :, ...] = [[[255, 0, 0]]]
        frames[i][:, :1, ...] = [[[255, 0, 0]]]
        frames[i][:, -1:, ...] = [[[255, 0, 0]]]

    return frames


def generate_instance_sequence(path):
    f_method = sorted(glob.glob(path + 'g*.png'))[-5:] + sorted(glob.glob(path + 'frnn_*.png'))
    return preprocess_predictions([ndim.imread(f) for f in f_method])


def build_sequences(name, paths):
    # Prepare sequences
    instances = [generate_instance_sequence(p) for p in paths]
    s_h, s_w = instances[0][0].shape[:2]

    # Prepare blank frames
    frame = 255 * np.ones((3*(s_h + 15) - 15, 8*(s_w + 15) - 15, 3), dtype=np.float32)
    frames = [np.copy(frame) for _ in range(15)]

    # Merge sequences
    for i, f in enumerate(zip(*instances)):
        for j, m in enumerate(f):
            l, t = (j % 8) * (s_w + 15), (j / 8) * (s_h + 15)
            frames[i][t:t+s_h, l:l+s_w] = m

    # Generate GIF
    imageio.mimsave(name, frames, duration=0.5)


if __name__ == '__main__':
    PATH_IN = '/home/moliu/Documents/Papers/Supplementary/images/qualitative/'
    PATH_OUT = '../'

    build_sequences(PATH_OUT + 'examples.gif', [
        PATH_IN + 'mmnist_l1/s12/',     PATH_IN + 'mmnist_l1/s11/',     PATH_IN + 'mmnist_l1/s13/',
        PATH_IN + 'mmnist_l1/s17/',     PATH_IN + 'mmnist_l1/s20/',     PATH_IN + 'mmnist_l1/s21/',
        PATH_IN + 'mmnist_l1/s11_n/',   PATH_IN + 'mmnist_l1/s5_n/',    PATH_IN + 'kth_l1/s31/',
        PATH_IN + 'kth_l1/s37/',        PATH_IN + 'kth_l1/s77/',        PATH_IN + 'kth_l1/s23/',
        PATH_IN + 'kth_l1/s43/',        PATH_IN + 'kth_l1/s75/',        PATH_IN + 'kth_l1/s97/',
        PATH_IN + 'kth_l1/s37_2/',      PATH_IN + 'ucf101_l1/s8/',      PATH_IN + 'ucf101_l1/s9_last/',
        PATH_IN + 'ucf101_l1/s9_mean/', PATH_IN + 'ucf101_l1/s21/',     PATH_IN + 'ucf101_l1/s37/',
        PATH_IN + 'ucf101_l1/s44/',     PATH_IN + 'ucf101_l1/s28/',     PATH_IN + 'ucf101_l1/s41/',
    ])
