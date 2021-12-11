import torch
from torch import nn
import numpy as np


def backpropagation(network, input_example):
    network.layers
    pass


# init weights should be a function that returns a single initialized weight
def generate_weights(init_weight_func, amount):
    weights = []
    for i in range(amount):
        weights.append(init_weight_func())
    return weights


class CustomNeuralNetwork:
    def __init__(self, num_layers, init_weights, layer_width) -> None:
        self.network = []
        self.num_layers = num_layers
        self.weights = init_weights()
        for i in range(num_layers):
            layer = []
            for j in range(layer_width):
                n = Neuron(generate_weights(init_weights, layer_width))
                layer.append(n)
            self.network.append(layer)

    def forward(self):
        pass


class Neuron:
    def __init__(self, weights):
        self.weights = weights


def initialize_weights_normal(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.normal(m.weight.data)
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
        self.layer1 = nn.Linear(w, w)
        self.layer2 = nn.Linear(w, w)
        self.layer3 = nn.Linear(w, 1)

        self.network = nn.Sequential(
            self.layer1(),
            self.layer2(),
            self.layer3()
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
        pass
