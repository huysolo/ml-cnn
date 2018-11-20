import numpy as np

from CNN.load_mnist import load_mnist
from CNN.model.initial_CNN import initial_CNN

train_data, train_label, valid_data, valid_label = load_mnist()
cnn = initial_CNN()
lr = [0.01, 0.001]
cnn.train(train_data, train_label, lr, epoch=2, batch_size=100)

res = cnn.predict(valid_data)
res = res.reshape(valid_label.shape)

print('Accuracy is: %f' % (np.sum(res == valid_label) / float(np.max(valid_label.shape))))
