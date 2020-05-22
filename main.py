import torch
import pandas as pd

from network import Network

data = pd.read_csv('data.csv', sep=';', names=['X', 'Y', 'TrueX', 'TrueY'])

train = torch.tensor(data.values, dtype=torch.float)
min = torch.min(train)
max = torch.max(train)

train = (train - min) / (max - min)
# print(train)

net = Network()
learning_rate = 1e-4
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), learning_rate)

for i in range(100):
    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    counter = 0
    error = 0
    for x in trainset:
        counter += 1
        input = x[:, [0, 1]]
        target = x[:, [2, 3]]
        
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        error = loss
        net.zero_grad()
        loss.backward()
        optimizer.step()
    print('[', i, ']', error)




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