import kaggle
import sklearn
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import LabelEncoder
from Neural_Networks.BackPropNetwork import initialize_weights_normal, initialize_weights_zero, generate_weights


def test_model(dataset, network, labels):
    accuracy = 0
    for idx, row in enumerate(dataset):
        int_row = np.array(list(float(r) for r in row))
        result = network(torch.tensor(int_row, dtype=torch.float)).item()
        print(result)
        r = 0
        if result < 0:
            r = 1
        else:
            r = 0
        if r == float(labels[idx]):
            accuracy += 1
    print('Error', 1 - accuracy / len(dataset))


class KaggleNetwork(nn.Module):
    def __init__(self, w, num_hidden) -> None:
        super().__init__()
        self.layer1 = nn.Linear(14, w)
        self.layer2 = nn.Linear(w, w)
        self.layer3 = nn.Linear(w, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.num_hidden = num_hidden

    def forward(self, inp) -> torch.Tensor:
        x = self.relu(self.layer1(inp))
        for i in range(self.num_hidden):
            x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# age,workclass,fnlwgt,education,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country,income>50K
dataset_income_train = []
with open('../Data/Kaggle_Income/train_final.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_income_train.append(terms)
enc = LabelEncoder()

# Convert numeric data that is currently a string to a number
for row in dataset_income_train:
    for i in range(len(row)):
        try:
            row[i] = int(row[i])
        except ValueError as ve:
            pass

data = np.array(dataset_income_train)[:, :-1]
train_labels = np.array(dataset_income_train)[:, -1]
rearranged = data[:, [0, 2, 4, 10, 11, 12, 1, 3, 5, 6, 7, 8, 9, 13]]
d = rearranged[:, 6:]
train_inputs = rearranged[:, :6]
for i in range(8):
    enc.fit(d[:, i])
    out = np.array(enc.transform(d[:, i])).reshape(-1, 1)
    train_inputs = np.append(train_inputs, out, axis=1)
# ID,age,workclass,fnlwgt,education,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country
dataset_income_test = []
with open('../Data/Kaggle_Income/test_final.csv', 'r') as file:
    for line in file:
        terms = line.strip().split(',')
        dataset_income_test.append(terms)

# Convert numeric data that is currently a string to a number
for row in dataset_income_test:
    for i in range(len(row)):
        try:
            row[i] = int(row[i])
        except ValueError as ve:
            pass

test_data = np.array(dataset_income_test)[:, 1:]
test_Ids = np.array(dataset_income_test)[:, 1]
t_rearranged = test_data[:, [0, 1, 3, 5, 9, 11, 12, 13, 2, 4, 6, 7, 8, 9, 10]]
d = t_rearranged[:, 6:]
test_inputs = t_rearranged[:, :6]
for i in range(8):
    enc.fit(d[:, i])
    out = np.array(enc.transform(d[:, i])).reshape(-1, 1)
    test_inputs = np.append(test_inputs, out, axis=1)


model = KaggleNetwork(14, 5)
model.apply(initialize_weights_normal)

optimizer = optim.Adam(model.parameters(), lr=0.001)
for idx, row in enumerate(train_inputs):
    int_row = np.array(list(float(r) for r in row))
    expected = np.array([float(train_labels[idx])])
    optimizer.zero_grad()
    inp = model(torch.tensor(int_row, dtype=torch.float))
    loss = nn.MSELoss()
    output = loss(inp, torch.tensor(expected, dtype=torch.float))
    output.backward()
    optimizer.step()

# test model accuracy
test_model(train_inputs, model, train_labels)
# test_model(test_inputs, model, test_labels)