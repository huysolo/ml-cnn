
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten,  MaxPooling2D, Conv2D
from keras.callbacks import TensorBoard
(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
def im2col(X, kernel_size):
    row_length = X.shape[0] - kernel_size[0] + 1
    col_length = X.shape[1] - kernel_size[1] + 1
    result = np.zeros((col_length * row_length, np.prod(kernel_size)))
    i = 0
    for row in range(0, row_length):
        for col in range(0, col_length):
            window = X[row:row+kernel_size[0], col:col+kernel_size[1]]
            result[i] = np.ndarray.flatten(window)
            i = i + 1
    return np.rot90(result), (row_length, col_length)
def kernel2row(kernel, X_size):
    print(X_size)
    row_length = X_size[0] - kernel.shape[0]
    col_length = X_size[1] - kernel.shape[1]
    rotated_kernel = np.rot90(kernel, 2)
    result = np.zeros(((col_length ) * (row_length ), np.prod(X_size)))
    print(result.shape)
    i = 0
    for row in range(0, row_length - 1):
        for col in range(0, col_length - 1):
            # print(((row, row_length - row), (col, col_length - col)))
            window =  np.pad(rotated_kernel, (((row, row_length - row), (col, col_length - col))), 'constant')
            # print(window.shape)
            # print(window.shape)
            result[i] = np.ndarray.flatten(window)
            i = i + 1
            # print(window.shape)
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
        # row_length = data.shape[0] - self.kernel.shape[0]
        # col_length = data.shape[1] - self.kernel.shape[1]
        # rotated_kernel = np.rot90(self.kernel, 2)
        # flat_rotated_kernel = np.ndarray.flatten(rotated_kernel)
        # self.final_result = np.zeros((row_length+1, col_length+1))
        # for row in range(0, row_length):
        #     for col in range(0, col_length):
        #         window = data[row:row+self.kernel_size[0],
        #                     col:col+self.kernel_size[1]]
        #         sub_result = np.ndarray.flatten(window @ rotated_kernel)
        #         self.final_result[row][col] = self.activation.forward(np.dot(sub_result, flat_rotated_kernel))
        self.X, data_shape = im2col(self.data, self.kernel_size)
        flatW =  np.ravel(self.kernel)
        Y = np.reshape(np.dot(flatW, self.X), data_shape)
        
        return self.activation.forward(Y)

    def backward(self, dy):
        # dW = np.reshape(dy, )
        flatY = self.activation.backward(np.ravel(dy))
        dW = np.reshape(np.dot(flatY, self.X.T), self.kernel.shape)
        self.kernel = np.subtract(self.kernel, dW)
        convW = kernel2row(self.kernel, self.data.shape)
        dX = np.reshape(np.dot(convW.T, flatY), self.X.shape)

        # rotated_data = np.rot90(self.data, 2)
        # rotated_dy = np.rot90(dy, 2)
        # d_ay = self.activation.backward(rotated_dy)

        # row_length = self.data.shape[0] - d_ay.shape[0]
        # col_length = self.data.shape[1] - d_ay.shape[1]

        # d_weight = np.zeros((row_length+1, col_length+1))
        # d_x = np.zeros(self.data.shape)

        # for row in range(0, row_length):
        #     for col in range(0, col_length):
        #         window = rotated_data[row:row+d_ay.shape[0],col:col+d_ay.shape[1]]
        #         sub_result = window @ d_ay
        #         d_weight[row][col] = np.sum(sub_result)
        # rotated_kernel = np.rot90(self.kernel, 2)
        
        # big_kernel_zeros = [x - 1 for x in dy.shape]
        # big_kernel = np.pad(rotated_kernel, (tuple(big_kernel_zeros), tuple(big_kernel_zeros)), 'constant')
        # row_length2 = big_kernel.shape[0] - dy.shape[0]
        # col_length2 = big_kernel.shape[1] - dy.shape[1]
        # for row in range(0, row_length2):
        #     for col in range(0, col_length2):
        #         window = big_kernel[row:row + d_ay.shape[0], col:col+d_ay.shape[1]]
        #         sub_result = window @ d_ay
        #         d_x[row][col] = np.sum(sub_result)
        # self.kernel = np.subtract(self.kernel, d_weight)

        return dX

class FC:
    def __init__(self, W_size=(10,26), eps = 0.01, activation = SIGMOID()):
        print(W_size[0])
        self.activation = activation
        self.W = eps * np.random.rand(W_size[0], W_size[1])
        self.eps = eps
        
    def forward(self, X):
        self.data = X
        out = self.W.dot(X)
        probs = self.activation.forward(out)
        return probs

    def backward(self, dZ):
        back = self.activation.backward(dZ)
        dW = np.dot(self.data, np.transpose(dZ))
        dX = np.dot(np.transpose(dZ), self.W)
        probs = dX
        
        self.W += -self.eps * np.transpose(dW)
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
                pic_loss = (train_label[index] - Y.T)**2
                if np.argmax(Y) == np.argmax(train_label[index]):
                    acc = acc + 1
                    print(acc)
                if (index+1)%100 == 0:
                    #print(str(epoch_idx) + ": " + str(index+1) + ": " + str(pic_loss.sum()))
                    print ("Loss: "+str(epoch_idx) + ": " + str(index+1) + ": " + str(np.mean(np.square(train_label[index] - Y.T))))
                dy = 2 * (Y.T - train_label[index]).T
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