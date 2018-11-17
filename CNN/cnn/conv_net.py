class ConvNet:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer_type, layer_params):
        pass

    def forward(self, data, label=[]):
        pass

    def backward(self, data, label, lr=0.01):
        pass

    def predict(self, test_data, batch_size=50):
        pass

    def train(self, train_data, train_label, lr, epoch=20, batch_size=100):
        pass
