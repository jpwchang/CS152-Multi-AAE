from glob import glob
from random import shuffle

from scipy.misc import imread
import numpy as np

def img_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def load_data(data_path):
    """
    Loads images from the given path and returns them as a training and
    testing set.
    """

    filenames = glob(data_path + "/*.png")
    shuffle(filenames)
    filenames = filenames * 50
    num_test = int(0.9 * len(filenames))
    train_filenames = filenames[:num_test]
    test_filenames = filenames[num_test:]

    train_data = []
    test_data = []
    for filename in train_filenames:
        train_data.append(imread(filename, mode='RGB'))
    for filename in test_filenames:
        test_data.append(imread(filename, mode='RGB'))

    train_matrix, test_matrix = np.stack(train_data, axis=0), np.stack(test_data, axis=0)
    return img_process(train_matrix), img_process(test_matrix)
