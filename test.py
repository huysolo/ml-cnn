from __future__ import division
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 * 2 / (9 + 12 + 2 * 31 + 333) # do the computation

# Host code   
data = numpy.ones(1000)
threadsperblock = 1000
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](data)
print(data)