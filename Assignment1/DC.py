import torch.nn as nn
import numpy as np

class DC(nn.Module):
    def __init__(self, features, pos_tags, layers=[]):
        super(DC, self).__init__()

        self.neg_slope = 0.01

        layers = [features] + layers + [pos_tags]
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i>0:
                modules.append(nn.LeakyReLU(self.neg_slope))
        
        self.net = nn.Sequential(*modules)
    
    def forward(self, x):
        out = self.net(x)

        return out