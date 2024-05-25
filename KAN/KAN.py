import torch.nn as nn
from KAN.KANLayer import MyKANLayer

class MyKAN(nn.Module):
    
    def __init__(self, width=None, grid=3, degree=3, approx_type="spline", seed=69, device='cpu'):
        """
        width: 
            list of integers, the width of each layer in the KAN
        grid: 
            int, the number of grid points to use for the spline
        degree: 
            int, the degree of the spline
        approx_type: "taylor" | "spline"
            string, the type of spline to use
        seed: 
            int, the seed for the random number generator
        device: 
            string, the device to run the KAN on

        Initializes the KAN with the given parameters
        """
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