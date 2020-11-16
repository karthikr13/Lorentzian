from wrapper import NetworkWrapper


class Train():
    def __init__(self, low, high, num_points, layer_sizes, num_osc, opt, decay):
        self.network = NetworkWrapper(low, high, num_points, layer_sizes, num_osc, opt, decay)
    def train(self, epochs):
        self.network.train(epochs)