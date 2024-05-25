import torch.nn as nn
from KAN.KANLayer import MyKANLayer

class MyKAN(nn.Module):
    
    def __init__(self, width=None, grid=3, degree=3, approx_type="taylor", seed=69, device='cpu'):
        super(MyKAN, self).__init__()

        # intialize variables for the KAN
        self.biases = []
        self.act_fun = nn.ModuleList()
        self.depth = len(width) - 1
        self.width = width

        # create the layers here
        for l in range(self.depth):
            kan_layer = MyKANLayer(width[l], width[l+1], grid, degree, approx_type, device=device)
            self.act_fun.append(kan_layer)

    # x should only be passed in batches
    def forward(self, x):

        # ensure the input is a vector
        # This means images are flattened
        x = x.view(x.size(0), -1)


        for l in range(self.depth):
            x = self.act_fun[l](x)
            
        return x
    
    def plot(self):
        for l in range(self.depth):
            self.act_fun[l].plot()