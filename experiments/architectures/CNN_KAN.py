import sys
sys.path.append("..")

from CKAN import *
from KAN import *
import torch.nn as nn


class CNNKANNet(nn.Module):
    """
    Architecture for a CKAN model with a normal Linear FF layer
    """
    def __init__(self, in_channels, hidden_channels, out_features, grid, degree, device='cuda:0'):
        super(CNNKANNet, self).__init__()
        self.conv1 =  nn.Conv2d(in_channels, hidden_channels[0],
                                kernel_size=3,
                                padding=1, device=device)
        self.conv2 =  nn.Conv2d(hidden_channels[0], hidden_channels[1],
                                kernel_size=3,
                                padding=1, device=device)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.linear1 = MyKANLayer(7*7*hidden_channels[1], out_features, 
                                    grid=grid, degree=degree, approx_type='spline', device=device)

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = self.max_pool(x)
        # Second convolutional layer
        x = self.conv2(x)
        x = self.max_pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Linear layer
        x = self.linear1(x)
        return x
  
    def update_grid(self, grid):
        self.linear1.update_grid(grid)