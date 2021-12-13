import torch
from torch import nn
import numpy as np


# computes the partial derivatives assuming loss function


# init weights should be a function that returns a single initialized weight
def generate_weights(init_weight_func, amount):
    weights = []
    for i in range(amount):
        weights.append(init_weight_func())
    return np.array(weights)


class CustomNeuralNetwork:
    def __init__(self, num_layers, init_weights, layer_width, activation_func) -> None:
        self.network = []
        self.output = 0
        self.num_layers = num_layers
        self.biases = [1] * num_layers
        for i in range(num_layers):
            layer = []
            for j in range(layer_width):
                n = Neuron(generate_weights(init_weights, layer_width), activation_func)
                layer.append(n)
            self.network.append(layer)

    def forward(self, input_X):
        for i in range(self.num_layers):
            layer = self.network[i]
            for neuron in layer:
                if neuron.prev is None:
                    neuron.input = input_X
                else:
                    neuron.input = neuron.prev.output
                neuron.output = neuron.activation_func(np.dot(neuron.input, neuron.weights))

    def backpropagation_compute_gradient(self, expected):
        grads = []
        for i, layer in reversed(list(enumerate(self.network))):
            for j, neuron in enumerate(layer):
                print(i, j)
                neuron.grad = (self.output - expected) * neuron.weights

        pass


class Neuron:
    def __init__(self, weights, activation_func):
        self.weights = weights
        self.input = None
        self.output = None
        self.grad = 0
        self.activation_func = activation_func
        self.prev = None
        self.next = None


def initialize_weights_normal(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 1)


def initialize_weights_zero(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.constant_(m.weight.data, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 1)


class SGDNeuralNetwork(nn.Module):
    def __init__(self, w) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, w)
        self.layer2 = nn.Linear(w, w)
        self.layer3 = nn.Linear(w, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, inp) -> torch.Tensor:
        x = self.tanh(self.layer1(inp))
        for i in range(9):
            x = self.tanh(self.layer2(x))
        x = self.layer3(x)
        return x
