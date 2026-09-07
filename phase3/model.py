import torch.nn as nn

'''
A new class that inherits from nn.Module 
(PyTorch's base class for all models). 
This provides access to features such as 
parameters(), train(), eval(), and to(device).
'''

class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64):
        # Call the parent class constructor (nn.Module) 
        # to correctly initialize the model.
        super().__init__()
        # first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        # second hidden layer
        self.fc2 = nn.Linear(hidden1, hidden2)
        # output layer (no activation)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        '''forward pass definition.'''
        # data through first linear layer + ReLU
        x = self.relu(self.fc1(x))
        # data through second linear layer + ReLU
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x