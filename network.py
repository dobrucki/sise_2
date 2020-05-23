import torch    
import torch.nn as nn

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()

        # Input layers
        self.input_layer_1 = nn.Linear(2, 3)

        # Hidden layers
        self.hidden_layer_1 = nn.Linear(3, 3)
        # self.hidden_layer_2 = nn.Linear(3, 3)
        # self.hidden_layer_3 = nn.Linear(3, 3)

        # Output layers
        self.output_layer_1 = nn.Linear(3, 2)


        
    def forward(self, x):
        
        # Input layers
        x = self.input_layer_1(x)

        # Hidden layers
        x = torch.sigmoid(self.hidden_layer_1(x))
        # x = torch.sigmoid(self.hidden_layer_2(x))
        # x = torch.sigmoid(self.hidden_layer_3(x))

        # Output layers
        x = self.output_layer_1(x)

        return x
