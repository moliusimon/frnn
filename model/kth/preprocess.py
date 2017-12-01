from scipy.misc import imresize
import numpy as np
import cv2
import h5py
import glob

# SCRIPT FOR PREPROCESSING THE KTH DATASET
#   This script prepares the training and test partitions for the KTH dataset. It will take the raw dataset from the
#   directory specified in DATA_PATH and place the processed partitions in SAVE_PATH.

DATA_PATH = '/mnt/DumperA/kth/'
SAVE_PATH = '/workmem/'


def partition_data(path):
    # List files and corresponding person IDs
    files = glob.glob(path + '*.avi')
    persons = np.array([int(f.split('/person')[1].split('_')[0]) for f in files])
    train_mask = persons <= 16

    return [files[i] for i in np.where(train_mask)[0]], [files[i] for i in np.where(~train_mask)[0]]


def prepare_sequence(file_path, height, width):
    vidcap = cv2.VideoCapture(file_path)

    # Read sequence frame by frame
    frames, (success, image) = [], vidcap.read()
    while success:
        image = imresize(image[:, 5:-5, 0], size=(height, width, 3), interp='bilinear')
        frames.append(image)
        success, image = vidcap.read()

    return np.expand_dims(np.stack(frames, axis=0), axis=3)


def prepare_partition(save_path, files, height, width):
    # List video files in directory, prepare HDF5 database
    database = h5py.File(save_path, 'w')

    # Create a dataset for each video file
    for i, f in enumerate(files):
        sequence = prepare_sequence(f, height, width)
        dataset = database.create_dataset(str(i), sequence.shape, dtype='u1')
        dataset[...] = sequence

    database.close()


if __name__ == '__main__':
    # Partition data by individuals
    files_train, files_test = partition_data(DATA_PATH)

    # Prepare train partition
    prepare_partition(
        save_path=SAVE_PATH + 'kth_train.hdf5',
        files=files_train,
        height=64,
        width=80,
    )

    # Prepare test partition
    prepare_partition(
        save_path=SAVE_PATH + 'kth_test.hdf5',
        files=files_test,
        height=64,
        width=80,
    )
