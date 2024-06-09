import sys
sys.path.append("..")

from CKAN import *
import torch.nn as nn


class CNNLINNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_features, device='cuda:0'):
        super(CNNLINNet, self).__init__()
        self.conv1 =  nn.Conv2d(in_channels, hidden_channels[0],
                                kernel_size=3,
                                padding=1, device=device)
        self.conv2 =  nn.Conv2d(hidden_channels[0], hidden_channels[1],
                                kernel_size=3,
                                padding=1, device=device)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*hidden_channels[1], out_features)

    def forward(self, x):
        # First convolutional layer
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x