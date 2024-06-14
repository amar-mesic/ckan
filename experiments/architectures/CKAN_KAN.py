import sys
sys.path.append("..")

from CKAN import *
from KAN import *
import torch.nn as nn


class CKANKANNet(nn.Module):
    """
    Architecture for a CKAN model with a normal Linear FF layer
    """
    def __init__(self, in_channels, hidden_channels, input_size, out_features, grid, degree, grid_range, device='cuda:0'):
        super(CKANKANNet, self).__init__()
        self.conv1 = CKANLayer(in_channels, hidden_channels[0],
                                kernel_size=3,
                                padding=1, grid=grid, degree=degree, grid_range=grid_range, device=device)
        self.conv2 = CKANLayer(hidden_channels[0], hidden_channels[1],
                                kernel_size=3,
                                padding=1, grid=grid, degree=degree, grid_range=grid_range, device=device)
        self.conv3 = CKANLayer(hidden_channels[1], hidden_channels[2],
                                kernel_size=3,
                                padding=1, grid=grid, degree=degree, grid_range=grid_range, device=device)
        self.max_pool = nn.MaxPool2d(2)
        final_dim = input_size // 8
        self.linear1 = MyKANLayer(final_dim*final_dim*hidden_channels[0], out_features, 
                                grid=grid, degree=degree, approx_type='spline', grid_range=grid_range, device=device)

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.max_pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Linear layer
        x = self.linear1(x)
        return x
  
    def update_grid(self, grid):
        self.conv1.update_grid(grid)
        self.conv2.update_grid(grid)
        self.conv3.update_grid(grid)