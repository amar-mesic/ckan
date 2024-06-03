import torch.nn as nn
import torch
from spline import *
from utils import *
import matplotlib.pyplot as plt
import spline




# creating a KAN layer from scratch
class MyKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, grid, degree=3, approx_type="taylor", grid_range=[-1, 1], device='cpu'):
        super(MyKANLayer, self).__init__()

        # initiliaze variables about the layer
        self.size = size = out_dim * in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.grid = grid
        self.grid_range = grid_range
        self.cache = None
        self.approx_type = approx_type

        # The spline function requires three parameters: knots, coeff, and degree
        # knots: the grid points for the spline
        self.knots = nn.Parameter(torch.linspace(grid_range[0], grid_range[1], steps=grid + 1, device=device).repeat(size, 1), requires_grad=False)

        # coeff: the coefficients for the spline - these are learnable!
        # I am wrapping them in a parameter since that is what they are
        self.coeff = nn.Parameter(0.1 * torch.randn(size, grid + degree if approx_type == "spline" else 1 + degree, device=device), requires_grad=True)


    def forward(self, x):

        # we process data in batches!
        batch_size = x.shape[0]

        # we need to repeat the input for each spline function
        # x.shape = (size, batch_size)
        x = x.transpose(0, 1).repeat(self.out_dim, 1)

        # store the input for later
        self.cache = x

        # store the output of the spline functions
        out = torch.zeros(self.size, batch_size)
        
        knots = self.knots
        coeff = self.coeff

        # print('x shape: ', x.shape)
        # print('knots shape: ', knots.shape)
        # print('coeff shape: ', coeff.shape)

        # calcuate the output of the spline functions
        if self.approx_type == "spline":
            spline_values = spline.coef2curve(x, knots, coeff, self.degree)
            out = spline_values

        else:
            # TODO: see if we can vectorize this
            for i in range(self.size):
                taylor_values = self.evaluate_taylor_series(x[i], coeff[i], self.degree)
                out[i] = taylor_values


        # reshape the output to be of shape (out_dim, in_dim, batch_size)
        # then we sum it as part of the algorithm
        # then we transpose it so subsequent layers can use it
        y = out.reshape(self.out_dim, self.in_dim, batch_size).sum(dim=1).transpose(0, 1)

        return y
    

    def update_grid(self, new_grid):
        x = self.cache if self.cache is not None else torch.zeros(self.size, 1)
        y = spline.coef2curve(x, self.knots, self.coeff, self.degree)
        # Update the grid points for the spline
        self.grid = new_grid
        self.knots.data = torch.linspace(self.grid_range[0], self.grid_range[1], steps=new_grid + 1, device=self.knots.device).repeat(self.size, 1)

        # self.coeff = nn.Parameter(0.1 * torch.randn(self.size, new_grid + self.degree if self.approx_type == "spline" else 1 + self.degree, device=self.coeff.device), requires_grad=True)
        self.coeff.data = spline.curve2coef(x, y, self.knots, self.degree)

        test = spline.coef2curve(x, self.knots, self.coeff, self.degree)
        # print(f"Old y: {y[0:5, 0]}\nNew y: {test[0:5, 0]}")
        # print("self.coeff.shape: ", self.coeff.shape)

    
    

    def evaluate_taylor_series(self, x, coeff, degree):
        # Evaluate the Taylor series of x using the coefficients
        exp = torch.arange(degree+1).view(1, -1).repeat(x.shape[0], 1)
        powed = torch.pow(x.view(-1, 1), exp)
        
        return torch.sum(coeff * powed, dim=1)


    # If we want to plot the spline curves of a layer
    def plot(self):
        # Plot the spline functions (optional)
        points = torch.linspace(self.grid_range[0], self.grid_range[1], 100)

        if self.approx_type == "spline":
            xs = points.repeat(self.size, 1)
            y = spline.coef2curve(xs, self.knots, self.coeff, self.degree).detach().cpu().numpy()
            for i in range(self.size):
                plt.plot(xs[i], y[i])
        else:
            for i in range(self.size):
                y = self.evaluate_taylor_series(points, coeff=self.coeff[i], degree=self.degree).detach().cpu().numpy()
                plt.plot(points, y, label=f'B_{i,3}(points)')
            
        plt.title('Cubic B-spline Basis Functions')
        plt.xlabel('x')
        plt.ylabel('B_{i,3}(points)')
        plt.legend()
        plt.show()
