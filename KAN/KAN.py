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
        self.grid = grid

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
    

    def update_grid(self, grid):
        for l in range(self.depth):
            self.act_fun[l].update_grid(grid)

    def train_model(self, n_epochs, train_iter, max_grid, step=2):
        min_grid = self.grid

        grid_update_freq = n_epochs // ((max_grid - min_grid) / step + 1)
        print("Grid update frequency: ", grid_update_freq)

        for epoch in n_epochs:
            if epoch % grid_update_freq == 0 and self.grid < max_grid:
                self.grid += step
                print(f"Updating grid to {self.grid} in epoch {epoch}")
                for l in range(self.depth):
                    self.act_fun[l].update_grid(self.grid)
                print("Grid updated to ", self.act_fun[0].grid)


            train_iter(epoch)
    


    def plot(self):
        for l in range(self.depth):
            self.act_fun[l].plot()