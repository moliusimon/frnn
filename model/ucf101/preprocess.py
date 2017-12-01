from scipy.misc import imresize
import numpy as np
import cv2
import h5py
import os
from random import shuffle

# SCRIPT FOR PREPROCESSING THE UCF-101 DATASET
#   This script prepares the training and test partitions for the UCF-101 dataset. It will take the raw dataset from the
#   directory specified in DATA_PATH and place the processed partitions in SAVE_PATH.

DATA_PATH = '/mnt/Storage/Datasets/action_recognition/UCF101/vid/'
SAVE_PATH = '/workmem/'


def prepare_sequence(file_path, height, width):
    # Read sequence frame by frame
    vidcap = cv2.VideoCapture(file_path)

    frames, (success, image) = [], vidcap.read()
    while success:
        image = imresize(image[..., [2, 1, 0]], size=(height, width, 3), interp='bilinear')
        frames.append(image)
        success, image = vidcap.read()

    return np.stack(frames, axis=0)


def prepare_partition(save_path, directory, partition, height, width):
    # List video files in directory, prepare HDF5 database
    files = [os.path.join(directory, f) for f in partition]
    database = h5py.File(save_path, 'w')

    # Create a dataset for each video file
    for i, f in enumerate(files):
        sequence = prepare_sequence(f, height, width)
        dataset = database.create_dataset(str(i), sequence.shape, dtype='u1')
        dataset[...] = sequence

    database.close()


def partition_data(path):
    # List files and group according to cateogry
    files, categories = [f for _, _, files in os.walk(path) for f in files if f.endswith('.avi')], {}
    for f in files:
        categories.setdefault(f.split('_')[1], []).append(f)

    # Partition each category using a 75%/25% split
    train, test = [], []
    for k in categories.keys():
        data = categories[k]
        shuffle(data)
        pivot = int(0.75 * len(data))
        train.extend(data[:pivot])
        test.extend(data[pivot:])

    return train, test

if __name__ == '__main__':
    # Partition data
    p_train, p_test = partition_data(DATA_PATH)

    # Prepare train partition
    prepare_partition(
        save_path=SAVE_PATH + 'ucf101_train.hdf5',
        directory=DATA_PATH,
        partition=p_train,
        height=64,
        width=85,
    )

    # Prepare test partition
    prepare_partition(
        save_path=SAVE_PATH + 'ucf101_test.hdf5',
        directory=DATA_PATH,
        partition=p_test,
        height=64,
        width=85,
    )
