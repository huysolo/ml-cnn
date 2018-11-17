import numpy as np
import _pickle
import gzip


def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = _pickle.load(f)
    f.close()
    train_data = train_set[0]
    train_data = np.reshape(train_data, (-1, 28, 28, 1))
    train_data = np.transpose(train_data, (1, 2, 3, 0)) * 255
    train_label = train_set[1]
    train_label = np.reshape(train_label, (1, -1))

    valid_data = valid_set[0]
    valid_data = np.reshape(valid_data, (-1, 28, 28, 1))
    valid_data = np.transpose(valid_data, (1, 2, 3, 0)) * 255
    valid_label = valid_set[1]
    valid_label = np.reshape(valid_label, (1, -1))

    return train_data, train_label, valid_data, valid_label
