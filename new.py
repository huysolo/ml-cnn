from keras.layers import Dense, Dropout
from keras.layers import Flatten,  MaxPooling2D, Conv2D
from keras.callbacks import TensorBoard
import numpy as np

(X_train,y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,28,28,1).astype('float32')
X_test = X_test.reshape(10000,28,28,1).astype('float32')

X_train /= 255
X_test /= 255

n_classes = 10


class Conv2D:
    def __init__(self, filter_number, kernel_size, activation, shape = None):
        self.filter_number = filter_number
        self.kernel_size = kernel_size
        self.activation = activation
        self.shape = shape

    def forward(self, data):
        if self.shape is not None:
            data = np.reshape(self.shape)
        weight = np.random.randn(data)