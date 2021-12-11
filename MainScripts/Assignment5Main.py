import numpy as np
from Neural_Networks.BackPropNetwork import SGDNeuralNetwork, initialize_weights_normal, initialize_weights_zero
from torch import optim
from torch import nn

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
model = SGDNeuralNetwork(5)
model.apply(initialize_weights_normal)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
for idx, row in enumerate(train_inputs):
    optimizer.zero_grad()
    output = model(row)
    loss = nn.MSELoss(output, train_labels[idx])
    loss.backward()
    optimizer.step()
