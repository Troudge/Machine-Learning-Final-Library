import numpy as np
from Neural_Networks.BackPropNetwork import SGDNeuralNetwork, initialize_weights_normal, initialize_weights_zero, \
    CustomNeuralNetwork, generate_weights
from torch import optim
from torch import nn
import torch


def init_weight_func():
    return np.random.normal()


def test_model(dataset, network, labels):
    accuracy = 0
    for idx, row in enumerate(dataset):
        result = network(torch.tensor(row, dtype=torch.float)).item()
        # print(result)
        r = 0
        if result >= 1:
            r = 1
        else:
            r = 0
        if r == labels[idx]:
            accuracy += 1
    print('Error', 1 - accuracy / len(dataset))


dataset_banknote = []
with open('../Data/bank-note/train.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        float_terms = [float(t) for t in terms]
        dataset_banknote.append(float_terms)

dataset_banknote_test = []
with open('../Data/bank-note/test.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        float_terms = [float(t) for t in terms]
        dataset_banknote_test.append(float_terms)

train_labels = np.array(dataset_banknote)[:, -1]
train_inputs = np.array(dataset_banknote)[:, :-1]

test_labels = np.array(dataset_banknote_test)[:, -1]
test_inputs = np.array(dataset_banknote_test)[:, :-1]

# section for my  implementation of the network. I was having trouble with this part, so it doesn't run without error
# the code for it is in backpropnetwork.py
# my_net = CustomNeuralNetwork(3, init_weight_func, 5, nn.Sigmoid)
# my_net.forward(train_inputs[1])
# my_net.backpropagation_compute_gradient(train_labels[1])

# a neural network using pytorch
model = SGDNeuralNetwork(15)
model.apply(initialize_weights_normal)

optimizer = optim.Adam(model.parameters(), lr=0.001)
for idx, row in enumerate(train_inputs):
    expected = np.array([train_labels[idx]])
    optimizer.zero_grad()
    inp = model(torch.tensor(row, dtype=torch.float))
    loss = nn.MSELoss()
    output = loss(inp, torch.tensor(expected, dtype=torch.float))
    output.backward()
    optimizer.step()

# test model accuracy
test_model(train_inputs, model, train_labels)
test_model(test_inputs, model, test_labels)
