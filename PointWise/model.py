import pandas as pd 
import numpy as np 
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm_notebook
import torch as t 

class myNet(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.fc1 = nn.Linear(136, 3*136)
    #     self.fc2 = nn.Linear(3*136, 3*64)
    #     #self.fc3 = nn.Linear(3*64, 3*64)
    #     self.fc4 = nn.Dropout(0.4)
    #     self.fc5 = nn.Linear(3*64, 1)
        
    # def forward(self, x):
    #     x = F.leaky_relu(self.fc1(x))
    #     x = F.leaky_relu(self.fc4(x))
    #     x = F.leaky_relu(self.fc2(x))
    #     #x = F.leaky_relu(self.fc4(x))
    #     #x = F.leaky_relu(self.fc3(x))
    #     x = F.leaky_relu(self.fc4(x))
    #     x = self.fc5(x)
    #     return x.squeeze()

    def __init__(self, n_feature, n_hidden_unit, n_hidden_layer, activation, dropout):
        super(myNet, self).__init__()
        self.layers = [nn.Linear(n_feature, n_hidden_unit)]
        for i in range(n_hidden_layer - 1):
            self.layers.append(activation())
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(n_hidden_unit, n_hidden_unit))
        self.layers.append(activation())
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(n_hidden_unit, 1))
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, feature):
        for module in self.layers:
            feature = module(feature)
        score = feature.squeeze()
        return score

    # def __init__(self, net_structures, leaky_relu=True, sigma=1.0, double_precision=False):
    #     super(myNet, self).__init__()
    #     self.fc_layers = len(net_structures)
    #     for i in range(len(net_structures) - 1):
    #         setattr(self, 'fc' + str(i + 1), nn.Linear(net_structures[i], net_structures[i+1]))
    #         if leaky_relu:
    #             setattr(self, 'act' + str(i + 1), nn.LeakyReLU())
    #         else:
    #             setattr(self, 'act' + str(i + 1), nn.ReLU())
    #     setattr(self, 'fc' + str(len(net_structures)), nn.Linear(net_structures[-1], 1))
    #     if double_precision:
    #         for i in range(1, len(net_structures) + 1):
    #             setattr(self, 'fc' + str(i), getattr(self, 'fc' + str(i)).double())
    #     self.sigma = sigma
    #     # self.activation = nn.Sigmoid()
    #     self.activation = nn.ReLU6()

    # def forward(self, input1):
    #     # from 1 to N - 1 layer, use ReLU as activation function
    #     for i in range(1, self.fc_layers):
    #         fc = getattr(self, 'fc' + str(i))
    #         act = getattr(self, 'act' + str(i))
    #         input1 = act(fc(input1))

    #     fc = getattr(self, 'fc' + str(self.fc_layers))
    #     return self.activation(fc(input1)) * self.sigma