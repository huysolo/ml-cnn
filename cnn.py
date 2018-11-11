import numpy
import gzip
from numba import cuda, float32

class ReLU:
    def forward(self, feature_map):
        return numpy.maximum(0, feature_map)
    def backward(self, pro,o_grad):
        print(pro)
        o_grad[pro < 0] = 0
        return o_grad
    
class CNN:
    def conv_(self, img, conv_filter):
        filter_size = conv_filter.shape[1]
        result = numpy.zeros((img.shape))
        #Looping through the image to apply the convolution operation.
        for r in numpy.uint16(numpy.arange(filter_size/2.0, 
                            img.shape[0]-filter_size/2.0+1)):
            for c in numpy.uint16(numpy.arange(filter_size/2.0, 
                                            img.shape[1]-filter_size/2.0+1)):
                """
                Getting the current region to get multiplied with the filter.
                How to loop through the image and get the region based on 
                the image and filer sizes is the most tricky part of convolution.
                """
                curr_region = img[r-numpy.uint16(numpy.floor(filter_size/2.0)):r+numpy.uint16(numpy.ceil(filter_size/2.0)), 
                                c-numpy.uint16(numpy.floor(filter_size/2.0)):c+numpy.uint16(numpy.ceil(filter_size/2.0))]
                #Element-wise multipliplication between the current region and the filter.
                curr_result = curr_region * conv_filter
                conv_sum = numpy.sum(curr_result) #Summing the result of multiplication.
                result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
                
        #Clipping the outliers of the result matrix.
        final_result = result[numpy.uint16(filter_size/2.0):result.shape[0]-numpy.uint16(filter_size/2.0), 
                            numpy.uint16(filter_size/2.0):result.shape[1]-numpy.uint16(filter_size/2.0)]
        return final_result
    
    def relu(self, feature_map):
    #Preparing the output of the ReLU activation function.
        relu_out = numpy.zeros(feature_map.shape)
        for map_num in range(feature_map.shape[-1]):
            for r in numpy.arange(0,feature_map.shape[0]):
                for c in numpy.arange(0, feature_map.shape[1]):
                    relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
        return relu_out
    def relu_derived(self, feature_map):
        relu_out = numpy.zeros(feature_map.shape)
        for map_num in range(feature_map.shape[-1]):
            for r in numpy.arange(0,feature_map.shape[0]):
                for c in numpy.arange(0, feature_map.shape[1]):
                    if feature_map[r, c, map_num] > 0:
                        relu_out[r, c, map_num] = 1
                    else:
                        relu_out[r, c, map_num] = 0

def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, numpy.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = numpy.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= numpy.max(vector)

    result = numpy.zeros(shape=(len(vector), num_classes))
    result[numpy.arange(len(vector)), vector] = 1
    return result.astype(int)

def getMnistData():
	with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
		data = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
	data = data.reshape(-1, 28, 28, 1)
	data = numpy.divide(data, 256)
	
	with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
		labels = numpy.frombuffer(f.read(), numpy.uint8, offset=8)
	labels = convertToOneHot(labels, 10)
	labels = labels.reshape(60000, 10, 1)
	
	with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
		testdata = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
	testdata = testdata.reshape(-1, 28, 28, 1)
	testdata = numpy.divide(testdata, 256)
	
	with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
		testlabels = numpy.frombuffer(f.read(), numpy.uint8, offset=8)
	testlabels = convertToOneHot(testlabels, 10)
	testlabels = testlabels.reshape(10000, 10, 1)	
	
	return data, labels, testdata, testlabels


dataset = getMnistData()
# print(dataset[0])

relu = ReLU()
print(dataset[0].shape)
o1_grad = numpy.zeros(dataset[0].shape)
# print(relu.backward(dataset[0][0], o1_grad))