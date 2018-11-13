import numpy as np
import gzip
import pickle
from time import time
from numba import cuda, float32
from keras.datasets import mnist


# epochs = 10
# neurons1 = 32
# neurons2 = 16
# learning_rate = 0.001

np.random.seed(1)
n_classes = 10
padding=1
stride=1

def saveWeights(wts):
    for num in range(len(wts)):
        with open('conv_weights/' + str(num + 1) + '.pkl', 'wb') as f:
            pickle.dump(wts[num], f, pickle.HIGHEST_PROTOCOL)


def loadWeight(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector) + 1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def getMnistData():

	(X_train,y_train), (X_test, y_test) = mnist.load_data()
	data = X_train.reshape(60000,28,28,1).astype('float32')
	testdata = X_test.reshape(10000,28,28,1).astype('float32')
	labels = convertToOneHot(y_train, n_classes)
	labels = labels.reshape(60000, n_classes, 1)
	testlabels = convertToOneHot(y_test, n_classes)
	testlabels = labels.reshape(60000, n_classes, 1)
	return data, labels, testdata, testlabels


def reLU(x):
    return np.maximum(0, x)


def reLU_back(pro, o_grad):
    o_grad[pro < 0] = 0
    return o_grad


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class CNN_CUDA:
    def __init__(self, epochs, neurons1, neurons2, learning_rate):
        self.epochs = epochs
        self.neurons1 = neurons1
        self.neurons2 = neurons2
        self.learning_rate = learning_rate

        weights = self.initWeights()
        self.f1, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = weights

    def conv_forward(self, pic, f, pro):
        cuda_conv_forward[(8), (26, 26)](pic, f, pro)

    def conv_backward(self, x, f, pro_grad, dx, f_grad):
        cuda_conv_backward(x, f, pro_grad, dx, f_grad)

    def pool_forward(self, o, p):
        cuda_pool_forward[(8), (13, 13)](o, p)

    def pool_backward(self, p_grad, o_grad):
        cuda_pool_backward(p_grad, o_grad)

    def initWeights(self):
        f1 = np.random.randn(8, 3, 3, 1) * np.sqrt(1.0 / 9)

        w1 = np.random.randn(self.neurons1, 1352) * np.sqrt(1.0 / 1352)
        b1 = np.zeros([self.neurons1, 1])

        w2 = np.random.randn(self.neurons2, self.neurons1) * np.sqrt(1.0 / self.neurons1)
        b2 = np.zeros([self.neurons2, 1])

        w3 = np.random.randn(10, self.neurons2) * np.sqrt(1.0 / self.neurons2)
        b3 = np.zeros([10, 1])

        return f1, w1, b1, w2, b2, w3, b3

    def rotate(self):
        mf1 = np.zeros(self.f1.shape)
        vf1 = np.zeros(self.f1.shape)
        mw1 = np.zeros(self.w1.shape)
        vw1 = np.zeros(self.w1.shape)
        mb1 = np.zeros(self.b1.shape)
        vb1 = np.zeros(self.b1.shape)
        mw2 = np.zeros(self.w2.shape)
        vw2 = np.zeros(self.w2.shape)
        mb2 = np.zeros(self.b2.shape)
        vb2 = np.zeros(self.b2.shape)
        mw3 = np.zeros(self.w3.shape)
        vw3 = np.zeros(self.w3.shape)
        mb3 = np.zeros(self.b3.shape)
        vb3 = np.zeros(self.b3.shape)

        return mf1, vf1, mw1, vw1, mb1, vb1, mw2, vw2, mb2, vb2, mw3, vw3, mb3, vb3

    def adam_optimization(self, dx, m, v):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = beta1 * m + (1 - beta1) * dx
        v = beta2 * v + (1 - beta2) * (dx ** 2)
        newWeights = self.learning_rate * m / (np.sqrt(v) + eps)
        return newWeights, m, v

	def conv_forward(self, X, W, b, stride=1, padding=1):
		cache = W, b, stride, padding
		n_filters, d_filter, h_filter, w_filter = W.shape
		n_x, d_x, h_x, w_x = X.shape
		h_out = (h_x - h_filter + 2 * padding) / stride + 1
		w_out = (w_x - w_filter + 2 * padding) / stride + 1

		if not h_out.is_integer() or not w_out.is_integer():
			raise Exception('Invalid output dimension!')

		h_out, w_out = int(h_out), int(w_out)

		X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
		W_col = W.reshape(n_filters, -1)

		out = W_col @ X_col + b
		out = out.reshape(n_filters, h_out, w_out, n_x)
		out = out.transpose(3, 0, 1, 2)

		cache = (X, W, b, stride, padding, X_col)

		return out, cache


	def conv_backward(self, dout, cache):
		X, W, b, stride, padding, X_col = cache
		n_filter, d_filter, h_filter, w_filter = W.shape

		db = np.sum(dout, axis=(0, 2, 3))
		db = db.reshape(n_filter, -1)

		dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
		dW = dout_reshaped @ X_col.T
		dW = dW.reshape(W.shape)

		W_reshape = W.reshape(n_filter, -1)
		dX_col = W_reshape.T @ dout_reshaped
		dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

		return dX, dW, db

    def train(self):

        mf1, vf1, mw1, vw1, mb1, vb1, mw2, vw2, mb2, vb2, mw3, vw3, mb3, vb3 = self.rotate()

        for epoch_no in range(0, self.epochs):

            num_correct = 0
            start = time()

            for index in range(0, 60000):
                o1 = np.zeros([26, 26, 8])
                pro1 = np.zeros([26, 26, 8])
                self.conv_forward(data[index], self.f1, pro1)
                o1 = reLU(pro1)
                p1 = np.zeros([13, 13, 8])
                self.pool_forward(o1, p1)
                pflat = p1.flatten().reshape(-1, 1)
                l1 = sigmoid(np.matmul(self.w1, pflat) + self.b1)
                l2 = sigmoid(np.matmul(self.w2, l1) + self.b2)
                l3 = sigmoid(np.matmul(self.w3, l2) + self.b3)

                if np.argmax(l3) == np.argmax(labels[index]):
                    num_correct += 1

                pic_loss = (labels[index] - l3) ** 2

                if (index + 1) % 100 == 0:
                    print(str(epoch_no) + ": " + str(index + 1) + ": " + str(pic_loss.sum()))

                l3_grad = 2 * (l3 - labels[index])
                sig3_grad = l3_grad * l3 * (1 - l3)
                b3_grad = sig3_grad
                wx3_grad = sig3_grad
                w3_grad = np.matmul(wx3_grad, np.transpose(l2))
                l2_grad = np.matmul(np.transpose(self.w3), wx3_grad)
                sig2_grad = l2_grad * l2 * (1 - l2)
                b2_grad = sig2_grad
                wx2_grad = sig2_grad
                w2_grad = np.matmul(wx2_grad, np.transpose(l1))
                l1_grad = np.matmul(np.transpose(self.w2), wx2_grad)
                sig1_grad = l1_grad * l1 * (1 - l1)
                b1_grad = sig1_grad
                wx1_grad = sig1_grad
                w1_grad = np.matmul(wx1_grad, np.transpose(pflat))
                pflat_grad = np.matmul(np.transpose(self.w1), wx1_grad)
                p_grad = pflat_grad.reshape(13, 13, 8)
                o1_grad = np.zeros([26, 26, 8])
                self.pool_backward(p_grad, o1_grad)

                pro1_grad = reLU_back(pro1, o1_grad)
                dx = np.zeros([28, 28, 1])
                f1_grad = np.zeros([8, 3, 3, 1])
                self.conv_backward(data[index], self.f1, pro1_grad, dx, f1_grad)

                f1_, mf1, vf1 = self.adam_optimization(f1_grad, mf1, vf1)
                self.f1 -= f1_
                w1_, mw1, vw1 = self.adam_optimization(w1_grad, mw1, vw1)
                self.w1 -= w1_
                b1_, mb1, vb1 = self.adam_optimization(b1_grad, mb1, vb1)
                self.b1 -= b1_
                w2_, mw2, vw2 = self.adam_optimization(w2_grad, mw2, vw2)
                self.w2 -= w2_
                b2_, mb2, vb2 = self.adam_optimization(b2_grad, mb2, vb2)
                self.b2 -= b2_
                w3_, mw3, vw3 = self.adam_optimization(w3_grad, mw3, vw3)
                self.w3 -= w3_
                b3_, mb3, vb3 = self.adam_optimization(b3_grad, mb3, vb3)
                self.b3 -= b3_

            print(str(epoch_no) + ": " + 'Training Accuracy: ' + str(num_correct / 600) + '%')
            print('Time taken: ' + str((time() - start)) + 's')
            wts = [self.f1, self.w1, self.b1, self.w2, self.b2, self.w3,
                   self.b3]
            saveWeights(wts)


@cuda.jit
def cuda_conv_forward(pic, f, pro):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    f_num = cuda.blockIdx.x

    fil = f[f_num]

    for i in range(3):
        for j in range(3):
            pro[tx, ty, f_num] += pic[tx + i, ty + j, f_num] * fil[i, j, 0]


@cuda.jit
def cuda_conv_backward(x, f, pro_grad, dx, f_grad):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    f_num = cuda.blockIdx.x

    fil = f[f_num]
    for i in range(3):
        for j in range(3):
            tmp1 = f[f_num, i, j, 0] * pro_grad[tx, ty, f_num]
            tmp2 = x[tx + i, ty + j, 0] * pro_grad[tx, ty, f_num]
            cuda.atomic.add(dx, (tx + i, ty + j, 0), tmp1)
            cuda.atomic.add(f_grad, (f_num, i, j, 0), tmp2)


@cuda.jit
def cuda_pool_forward(o, p):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    d = cuda.blockIdx.x

    p[tx, ty, d] = max(o[2 * tx, 2 * ty, d], o[2 * tx + 1, 2 * ty, d], o[2 * tx, 2 * ty + 1, d],
                       o[2 * tx + 1, 2 * ty + 1, d])


@cuda.jit
def cuda_pool_backward(p_grad, o_grad):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    d = cuda.blockIdx.x

    o_grad[tx, ty, d] = p_grad[int(tx / 2), int(ty / 2), d]


# Prepare data
data, labels, testdata, testlabels = getMnistData()


def loadWeights():
    f1 = loadWeight('conv_weights/1.pkl')
    w1 = loadWeight('conv_weights/2.pkl')
    b1 = loadWeight('conv_weights/3.pkl')
    w2 = loadWeight('conv_weights/4.pkl')
    b2 = loadWeight('conv_weights/5.pkl')
    w3 = loadWeight('conv_weights/6.pkl')
    b3 = loadWeight('conv_weights/7.pkl')
    return f1, w1, b1, w2, b2, w3, b3


CNN_C = CNN_CUDA(epochs=10, neurons1=32, neurons2=16, learning_rate=0.001)
CNN_C.train()

num_correct = 0
for index in range(0, 10000):
    o1 = np.zeros([26, 26, 8])
    pro1 = np.zeros([26, 26, 8])
    CNN_C.conv_forward(testdata[index], CNN_C.f1, pro1)
    o1 = reLU(pro1)
    p1 = np.zeros([13, 13, 8])
    CNN_C.pool_forward(o1, p1)
    pflat = p1.flatten().reshape(-1, 1)
    l1 = sigmoid(np.matmul(CNN_C.w1, pflat) + CNN_C.b1)
    l2 = sigmoid(np.matmul(CNN_C.w2, l1) + CNN_C.b2)
    l3 = sigmoid(np.matmul(CNN_C.w3, l2) + CNN_C.b3)
    if np.argmax(l3) == np.argmax(testlabels[index]):
        num_correct += 1
    if (index + 1) % 100 == 0:
        print(str(index + 1))

print("Test Accuracy: " + str(num_correct / 100) + '%')
