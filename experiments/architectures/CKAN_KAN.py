import sys
sys.path.append("..")

from CKAN import *
from KAN import *
import torch.nn as nn


class CKANKANNet(nn.Module):
    """
    Architecture for a CKAN model with a normal Linear FF layer
    """
    def __init__(self, in_channels, hidden_channels, fc_dim, out_features, grid, degree, device='cuda:0'):
        super(CKANKANNet, self).__init__()
        self.conv1 = CKANLayer(in_channels, hidden_channels[0],
                                kernel_size=3,
                                padding=1, grid=grid, degree=degree, device=device)
        self.max_pool = nn.MaxPool2d(2)
        self.linear1 = MyKANLayer(fc_dim*fc_dim*hidden_channels[0], out_features, 
                                grid=grid, degree=degree, approx_type='spline', device=device)

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = self.max_pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Linear layer
        x = self.linear1(x)
        return x
  
    def update_grid(self, grid):
        self.conv1.update_grid(grid)