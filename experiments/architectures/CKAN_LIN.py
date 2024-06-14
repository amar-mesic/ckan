import sys
sys.path.append("..")

from CKAN import *
import torch.nn as nn


class CKANLINNet(nn.Module):
    """
    Architecture for a CKAN model with a normal Linear FF layer
    """
    def __init__(self, in_channels, hidden_channels, input_size, out_features, grid, degree, grid_range, device='cuda:0'):
            super(CKANLINNet, self).__init__()
            self.conv1 = CKANLayer(in_channels, hidden_channels[0],
                                    kernel_size=3,
                                    padding=1, grid=grid, degree=degree, grid_range=grid_range, device=device)
            self.bn1 = nn.BatchNorm2d(hidden_channels[0])
          
            self.conv2 = CKANLayer(hidden_channels[0], hidden_channels[1],
                                  kernel_size=3,
                                  padding=1, grid=grid, degree=degree, grid_range=grid_range, device=device)
            self.bn2 = nn.BatchNorm2d(hidden_channels[1])
            
            self.conv3 = CKANLayer(hidden_channels[1], hidden_channels[2],
                                  kernel_size=3,
                                  padding=1, grid=grid, degree=degree, grid_range=grid_range, device=device)
            self.bn3 = nn.BatchNorm2d(hidden_channels[2])

            self.max_pool = nn.MaxPool2d(2)
            final_dim = input_size // 8
            self.dropout = nn.Dropout(0.2)
            self.linear1 = nn.Linear(final_dim*final_dim*hidden_channels[2], out_features)

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Linear layer
        x = self.linear1(x)
        return x
  
    def update_grid(self, grid):
        self.conv1.update_grid(grid)
        self.conv2.update_grid(grid)
        self.conv3.update_grid(grid)