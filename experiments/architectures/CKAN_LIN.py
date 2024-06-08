import sys
sys.path.append("..")

from CKAN import *
import torch.nn as nn


class CKANNet(nn.Module):
    """
    Architecture for a CKAN model with a normal Linear FF layer
    """
    def __init__(self, channels, out_features, grid, device='cuda:0'):
          super(CKANNet, self).__init__()
          self.conv1 = CKANLayer(channels[0], channels[1],
                                  kernel_size=3,
                                  padding=1, grid=grid, device=device)
          self.conv2 = CKANLayer(channels[1], channels[2],
                                kernel_size=3,
                                padding=1, grid=grid, device=device)
          self.max_pool = nn.MaxPool2d(2)
          self.linear1 = nn.Linear(7*7*channels[-1], out_features)

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
        self.conv1.update_grid(grid)
        self.conv2.update_grid(grid)