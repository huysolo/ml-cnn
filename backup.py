
import numpy as np
#from numbapro import vectorize
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten,  MaxPooling2D, Conv2D
from keras.callbacks import TensorBoard

import os
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/lib/nvidia-cuda-toolkit/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/lib/x86_64-linux-gnu/libnvvm.so"
(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
#@vectorize()
#def im2colcuda(X, kernel_size):

def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def im2col(X, kernel_size):
    row_length = X.shape[0] - kernel_size[0] + 1
    col_length = X.shape[1] - kernel_size[1] + 1
    result = np.zeros((col_length * row_length, np.prod(kernel_size)))
    i = 0
    # threadsperblock = (16, 16)
    # blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    # blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    for row in range(0, row_length):
        for col in range(0, col_length):
            num = row * row_length + col
            window = X[row:row+kernel_size[0], col:col+kernel_size[1]]
            result[num] = np.ndarray.flatten(window)
    return np.rot90(result), (row_length, col_length)
def kernel2row(kernel, X_size):
    row_length = X_size[0] - kernel.shape[0]
    col_length = X_size[1] - kernel.shape[1]
    rotated_kernel = np.rot90(kernel, 2)
    result = np.zeros(((col_length + 1) * (row_length + 1), np.prod(X_size)))
    i = 0
    for row in range(0, row_length):
        for col in range(0, col_length):
            window =  np.pad(rotated_kernel, (((row, row_length - row), (col, col_length - col))), 'constant')
            result[i] = np.ndarray.flatten(window)
            i = i + 1
    return result
def kernel2rowOp(kernel, X_size):
    row_diff = X_size[0] - kernel.shape[0]
    col_diff = X_size[1] - kernel.shape[1]
    row_length = kernel.shape[0] + 2 * row_diff
    col_length = kernel.shape[1] + 2 * col_diff
    rotated_kernel = np.rot90(kernel, 2)
    result = np.zeros(((X_size[0] - kernel.shape[0] + 1) * (X_size[1] - kernel.shape[1] + 1), np.prod(X_size)))

    big_kernel = np.pad(rotated_kernel, ((row_diff, row_diff), (col_diff, col_diff)), 'constant')
    for row in range(row_length, row_diff + 2, -1):
        for col in range(col_length, col_diff + 2, -1):
            num = (row_length - row_diff - 2) * (row_length - row) + (col_length - col)
            window = big_kernel[row - X_size[0] : row , col - X_size[1] : col ]
            result[num] = np.ndarray.flatten(window)
    return result

class FLATTEN:
    def forward(self, data):
        self.data = data
        return np.ndarray.flatten(data)
    def backward(self, dy):
        return np.reshape(dy, tuple(self.data.shape))
class RELU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, x):
        return self.mask


class SIGMOID:
    def forward(self, x):
        self.sig = 1. / (1. + np.exp(-x))
        return self.sig

    def backward(self, sig):
        return sig * (1. - sig)


class CONV:
    def __init__(self, kernel_size=(3, 3), activation='sigmoid'):
        if activation is 'sigmoid':
            self.activation = SIGMOID()
        if activation is 'relu':
            self.activation = RELU()  
        
        self.kernel_size = kernel_size
        self.kernel = np.random.rand(kernel_size[0], kernel_size[1])

    def forward(self, data):
        self.data = data
        self.X, data_shape = im2col(self.data, self.kernel_size)
        flatW =  np.ravel(self.kernel)
        Y = np.reshape(np.dot(flatW, self.X), data_shape)
        
        return self.activation.forward(Y)

    def backward(self, dy):
        flatY = self.activation.backward(np.ravel(dy))
        dW = np.reshape(np.dot(flatY, self.X.T), self.kernel.shape)
        self.kernel = np.subtract(self.kernel, dW)
        convW = kernel2rowOp(self.kernel, self.data.shape)
        dX = np.reshape(np.dot(convW.T, flatY), self.data.shape)

        return dX

class FC:
    def __init__(self, W_size=(10,26), eps = 0.01, activation = SIGMOID()):
        print(W_size[0])
        self.activation = activation
        self.W = eps * np.random.rand(W_size[0], W_size[1])
        self.b = np.random.randn(W_size[1]).reshape(1, W_size[1])
        self.eps = eps
        
    def forward(self, X):
        self.data = X
        out = self.W.dot(X)
        probs = self.activation.forward(out)
        return probs

    def backward_without_bias(self, dZ):        
        back_dZ=self.activation.backward(dZ)
        dW = np.dot(back_dZ,self.data)
        dX = np.dot(np.transpose(self.W),back_dZ)
        probs=dX
        
        self.W += -self.eps * dW
        return probs
      
    def backward(self, dZ):        
        back_dZ=self.activation.backward(dZ)
        back_bias = back_dZ * np.ones_like(back_dZ)
        
        db = np.dot(np.ones((1, back_dZ.shape[0]), dtype=np.float64), back_dZ)
        dW = np.dot(back_bias,self.data)
        dX = np.dot(np.transpose(self.W),back_bias)
        probs=dX
        
        self.b += -self.eps * db
        self.W += -self.eps * dW
        return probs
      
class Softmax:
    def predict(self,X):
        exp_scores = np.exp(X)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self,X,y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        return 1./num_examples * data_loss

    def diff(self,X,y):
        num_examples = X.shape[0]
        probs = self.predict(X)       
        probs[range(num_examples), y] -= 1
        return probs
class CNN:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def train(self, train_data, train_label, lr, epoch=20):
        softmaxOutput = Softmax()
        for epoch_idx in np.arange(epoch):
            acc = 0
            for index in range(0, train_data.shape[0]):
                Y = train_data[index]
                for layer in self.layers:
                    Y = layer.forward(Y)
                diff_loss = softmaxOutput.diff(Y, train_label[index].astype(int))
                if (index+1)%100 == 0:
                    print ("accuracy: "+str(epoch_idx) + ": " + str(index+1) + ": " + str(np.mean(np.square(train_label[index] - Y.T))))
                dy = diff_loss
                for layer in self.layers[::-1]:
                    dy = layer.backward(dy)
   
cnn = CNN()
cnn.add(CONV((3, 3), 'sigmoid'))
#cnn.add(FLATTEN())
cnn.add(FC(W_size=(10, 26)))
cnn.train(X_train, y_train, 0.01, 1)
kernel = np.random.rand(3, 3)
# x = im2col(X_train[0], kernel.shape)
# w = kernel2row(kernel, X_train[0].shape)
# y = np.matmul(x, w)
# print(y.shape)

# print(im2col_indices(X_train[0], 3, 3, 0))