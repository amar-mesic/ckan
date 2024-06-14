import sys
sys.path.append("..")

from CKAN import *
import torch.nn as nn

class CNNLINNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, input_size, out_features, device='cuda:0'):
        super(CNNLINNet, self).__init__()
        
        # Initialize convolutional layers
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1, device=device)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, padding=1, device=device)
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])
        
        self.conv4 = nn.Conv2d(hidden_channels[2], hidden_channels[3], kernel_size=3, padding=1, device=device)
        self.bn4 = nn.BatchNorm2d(hidden_channels[3])
        
        self.conv5 = nn.Conv2d(hidden_channels[3], hidden_channels[4], kernel_size=3, padding=1, device=device)
        self.bn5 = nn.BatchNorm2d(hidden_channels[4])
        
        self.conv6 = nn.Conv2d(hidden_channels[4], hidden_channels[5], kernel_size=3, padding=1, device=device)
        self.bn6 = nn.BatchNorm2d(hidden_channels[5])
        
        # Initialize other layers
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.2)
        final_dim = input_size // 8  # Calculate dimension for linear layer input
        self.linear1 = nn.Linear(final_dim * final_dim * hidden_channels[5], out_features)

    def forward(self, x):
        # First set of layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        # Second set of layers
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        # Third set of layers
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        # Flatten and pass through linear layer
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x
