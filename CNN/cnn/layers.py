import numpy as np


def conv_forward(x, w, b, params):
    pass


def conv_backward(x, w, b, conv_param, dout):
    pass


def max_pooling_forward(x, pool_params):
    pass


def max_pooling_backward(x, dout, pool_params):
    pass


def relu_forward(x):
    out = np.where(x > 0, x, 0)
    return out


def relu_backward(x, dout):
    dx = np.where(x > 0, dout, 0)
    return [dx]
