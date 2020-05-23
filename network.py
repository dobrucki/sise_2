import torch    
import torch.nn as nn

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc3 = nn.Linear(4, 2)

        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc3(x)
        return x
