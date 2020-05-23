import torch
import pandas as pd
from statistics import mean

from network import Network

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

data = pd.read_csv('data.csv', sep=';', names=['X', 'Y', 'TrueX', 'TrueY'])


train = torch.tensor(data.values, dtype=torch.float)
min = torch.min(train)
max = torch.max(train)

train = (train - min) / (max - min)
# train = train.normal(0, 1)
# print(train.get_device())



net = Network()
learning_rate = 1e-2
criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), learning_rate)

errors = []
for i in range(10000):
    trainset = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True)
    counter = 0
    optimizer = torch.optim.SGD(net.parameters(), learning_rate)
    for x in trainset:
        counter += 1
        input = x[:, [0, 1]]
        target = x[:, [2, 3]]
        
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        errors.append(loss.item()) 
        net.zero_grad()
        loss.backward()
        optimizer.step()
    learning_rate *= .9
    print('[', i, ']', 'average error: ', mean(errors))




trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)    
for x in trainset:
    input = x[:, [0, 1]]
    check = x[:, [2, 3]]
    prediction = net(input)
    print(prediction)
    print(check)
    break
# # true_set = torch.tensor(data[['TrueX', 'TrueY']].values, dtype=torch.float)
# # print(true_set)
# # pred = net(training_set)
# # loss = loss_fn(pred, true_set)
# # print(loss.item())

# # optimizer.zero_grad()
# # loss.backward()
# # optimizer.step()