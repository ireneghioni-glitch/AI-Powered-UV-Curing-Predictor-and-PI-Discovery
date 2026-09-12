import torch.nn as nn

'''
A new class that inherits from nn.Module 
(PyTorch's base class for all models). 
This provides access to features such as 
parameters(), train(), eval(), and to(device).
'''

class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout_rate=0.0):
        # Call the parent class constructor (nn.Module) 
        # to correctly initialize the model.
        super().__init__()
        # first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        # adding droput (droput_rate=0.0, 0% of the neurons are randomly deactivated)
        # once real data are avaliable, try 0.1, 0.2 and 0.3 
        # IMPORTANT! I will need also to increase epochs num and introduce Validation set
        self.dropout = nn.Dropout(dropout_rate)
        # second hidden layer
        self.fc2 = nn.Linear(hidden1, hidden2)
        # output layer (no activation)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        '''forward pass definition.'''
        # data through first linear layer + ReLU
        x = self.relu(self.fc1(x))
        # apply droput after activation
        x = self.dropout(x)
        # data through second linear layer + ReLU
        x = self.relu(self.fc2(x))
        # apply droput after activation
        x = self.dropout(x)
        x = self.out(x)
        return x