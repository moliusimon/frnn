import imageio
import glob
import scipy.ndimage as ndim
import scipy.misc as sm
import numpy as np

# Prepare method strings
PATH_STRINGS = '/home/moliu/Documents/Papers/Supplementary/titles/'
s_titles = [
    ndim.imread(PATH_STRINGS + 'frnn.png'),
    ndim.imread(PATH_STRINGS + 'rladder.png'),
    ndim.imread(PATH_STRINGS + 'prednet.png'),
    ndim.imread(PATH_STRINGS + 'srivastava.png'),
    ndim.imread(PATH_STRINGS + 'mathieu.png'),
    ndim.imread(PATH_STRINGS + 'villegas.png'),
]


def generate_captions(strings, width):
    titles = [255 * np.ones((20, width+15, 3), dtype=np.uint8) for _ in strings]

    # Prepare strings
    strings = [(np.stack([s, s, s], axis=2) if len(s.shape) == 2 else s) for s in strings]
    for i, s in enumerate(strings):
        s = sm.imresize(s, size=0.8)
        t_pad, l_pad = (20 - s.shape[0]) / 2, (width - s.shape[1]) / 2
        titles[i][t_pad:t_pad+s.shape[0], l_pad:l_pad+s.shape[1]] = s

    return np.pad(
        np.concatenate(titles, axis=1)[:, :-15],
        ((0, 0), (width+15, 0), (0, 0)),
        'constant', constant_values=255
    )


def preprocess_predictions(gt, predictions, width=80):
    # Append dimension if no channels
    gt = [(f if len(f.shape) == 3 else np.expand_dims(f, axis=2)) for f in gt]
    predictions = [[(f if len(f.shape) == 3 else np.expand_dims(f, axis=2)) for f in s] for s in predictions]

    # Replicate channel if grayscale
    gt = [(f if f.shape[2] == 3 else np.concatenate([f, f, f], axis=2)) for f in gt]
    predictions = [[(f if f.shape[2] == 3 else np.concatenate([f, f, f], axis=2)) for f in s] for s in predictions]

    # Reshape predictions to gt shape, prepare black frames, fill leading frames
    predictions = [[sm.imresize(f, gt[0].shape[:2]) for f in s] for s in predictions]
    predictions = [([np.zeros(gt[0].shape, dtype=np.uint8)] * 10 if len(s) == 0 else s) for s in predictions]
    predictions = [[gt[4]] * 5 + s for s in predictions]

    # Pad frames to fit expected width
    padding = ((0, 0), ((width - gt[0].shape[1]) / 2,)*2, (0, 0))
    gt = [np.pad(f, padding, 'constant', constant_values=255) for f in gt]
    predictions = [[np.pad(f, padding, 'constant', constant_values=255) for f in s] for s in predictions]

    return gt, predictions


def generate_instance_sequence(path):
    # List ground truth images
    f_gt = sorted(glob.glob(path + 'g*.png'))
    f_gt = f_gt[-5:] + f_gt[:10]

    # List prediction images
    f_methods = [
        sorted(glob.glob(path + 'frnn_*.png')),     sorted(glob.glob(path + 'rladder_*.png')),
        sorted(glob.glob(path + 'prednet_*.png')),  sorted(glob.glob(path + 'srivastava_*.png')),
        sorted(glob.glob(path + 'mathieu_*.png')),  sorted(glob.glob(path + 'villegas_*.png'))
    ]

    # Read & preprocess frames
    f_gt, f_methods = [ndim.imread(f) for f in f_gt], [[ndim.imread(f) for f in m] for m in f_methods]
    f_gt, f_methods = preprocess_predictions(f_gt, f_methods)
    im_h, im_w = f_gt[0].shape[:2]

    # Fill frames with ground truth & predictions
    frame = 255 * np.ones((im_h, im_w*7 + 15*6, 3), dtype=np.uint8)
    frames = [np.copy(frame) for _ in range(15)]
    for i, (fg, fm) in enumerate(zip(f_gt, zip(*f_methods))):
        frames[i][:im_h, :im_w] = fg
        for j, (f, title) in enumerate(zip(fm, ['frnn', 'rladder', 'prednet', 'Srivastava', 'mathieu', 'villegas'])):
            r = (j + 1) * (im_w + 15)
            frames[i][:im_h, r:r+im_w] = f

    # Return sequence frames
    return frames


def build_dataset(name, paths):
    titles = generate_captions(s_titles, 80)
    instances = [generate_instance_sequence(p) for p in paths]
    s_h, s_w = instances[0][0].shape[:2]

    # Merge sequences
    frame = 255 * np.ones((len(paths) * (s_h + 15) - 15, s_w, 3), dtype=np.float32)
    frames = [np.copy(frame) for _ in range(15)]
    for i, f in enumerate(zip(*instances)):
        for j, m in enumerate(f):
            t = j*(s_h+15)
            frames[i][t:t+s_h] = m

    imageio.mimsave(name, [np.concatenate((titles, f), axis=0) for f in frames], duration=0.5)


if __name__ == '__main__':
    PATH_IN = '/home/moliu/Documents/Papers/Supplementary/images/qualitative/'
    PATH_OUT = '/home/moliu/Documents/Papers/Supplementary/gifs/'

    build_dataset(PATH_OUT + 'mmnist.gif', [
        PATH_IN + 'mmnist_l1/s12/',     PATH_IN + 'mmnist_l1/s11/',     PATH_IN + 'mmnist_l1/s13/',
        PATH_IN + 'mmnist_l1/s17/',     PATH_IN + 'mmnist_l1/s20/',     PATH_IN + 'mmnist_l1/s21/',
        PATH_IN + 'mmnist_l1/s11_n/',   PATH_IN + 'mmnist_l1/s5_n/',
    ])

    build_dataset(PATH_OUT + 'kth.gif', [
        PATH_IN + 'kth_l1/s31/',        PATH_IN + 'kth_l1/s37/',        PATH_IN + 'kth_l1/s77/',
        PATH_IN + 'kth_l1/s23/',        PATH_IN + 'kth_l1/s43/',        PATH_IN + 'kth_l1/s75/',
        PATH_IN + 'kth_l1/s97/',        PATH_IN + 'kth_l1/s37_2/',
    ])

    build_dataset(PATH_OUT + 'ucf101.gif', [
        PATH_IN + 'ucf101_l1/s8/',      PATH_IN + 'ucf101_l1/s9_last/', PATH_IN + 'ucf101_l1/s9_mean/',
        PATH_IN + 'ucf101_l1/s21/',     PATH_IN + 'ucf101_l1/s37/',     PATH_IN + 'ucf101_l1/s44/',
        PATH_IN + 'ucf101_l1/s28/',     PATH_IN + 'ucf101_l1/s41/',
    ])
