import torch
import pandas as pd
from statistics import mean
from math import fabs

from network import Network

data = pd.read_csv('data.csv', sep=';', names=['X', 'Y', 'TrueX', 'TrueY'])
test_data = pd.read_csv('test_data.csv', sep=';', names=['X', 'Y', 'TrueX', 'TrueY'])


train = torch.tensor(data.values, dtype=torch.float)
test = torch.tensor(test_data.values, dtype=torch.float, requires_grad=False)
min = torch.min(train)
max = torch.max(train)
test_min = torch.min(test)
test_max = torch.max(test)

train = (train - min) / (max - min)
test = (test - test_min) / (test_max - test_min)


net = Network()
learning_rate = 1
criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), learning_rate)

errors = []
for i in range(200):
    trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), learning_rate)
    # optimizer = torch.optim.Adam(net.parameters())
    for x in trainset:
        input = x[:, [0, 1]]
        target = x[:, [2, 3]]   
    
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        errors.append(loss.item()) 
        net.zero_grad()
        loss.backward()
        optimizer.step()
        learning_rate *= (1. - 1e-6)
    if i % 10 == 0:
        print('[', i, ']', 'mean error: ', mean(errors), ' | current learning rate: ', learning_rate)

testset = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)
error_fn = torch.nn.MSELoss()
errors = []
counter = 0
results = []
for x in testset:
    counter += 1
    input = x[:, [0, 1]]
    target = x[:, [2, 3]]
    output = net(input)
    input = input * (test_max - test_min) + test_min
    target = target * (test_max - test_min) + test_min
    output = output * (test_max - test_min) + test_min
    row = [
        input[0][0].item(), input[0][1].item(), target[0][0].item(), target[0][1].item(), output[0][0].item(), 
        output[0][1].item(), (target[0][0].item() - output[0][0].item()) ** 2 + (target[0][1].item() - output[0][1].item()) ** 2
    ]
    results.append(row)

results = pd.DataFrame(results, columns=['X', 'Y', 'TargetX', 'TargetY', 'OutputX', 'OutputY', 'Error'])
print(results.head(10))