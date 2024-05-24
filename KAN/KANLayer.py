import torch.nn as nn
import torch
import numpy as np
from spline import *
from utils import *
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt




# creating a KAN layer from scratch
class MyKANLayer(nn.Module):

    def __init__(self, in_dim, out_dim, grid, degree=3, approx_type="taylor", grid_range=[-1, 1], device='cpu'):
        super(MyKANLayer, self).__init__()

        # initiliaze variables about the layer
        self.size = size = out_dim * in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.grid_range = grid_range
        self.cache = None
        self.approx_type = approx_type

        # The spline function requires three parameters: knots, coeff, and degree
        # knots: the grid points for the spline
        # self.knots = nn.Parameter(torch.linspace(grid_range[0], grid_range[1], steps=grid + 1, device=device).repeat(size, 1)).requires_grad_(False)
        self.knots = nn.Parameter(torch.linspace(grid_range[0], grid_range[1], steps=grid + 1, device=device)).requires_grad_(False)

        # coeff: the coefficients for the spline - these are learnable!
        # I am wrapping them in a parameter since that is what they are
        # grid + 
        self.coeff = nn.Parameter(0.1 * torch.randn(size, 1 + degree, device=device)).requires_grad_(True)


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


        # OLD WAY: WITH BUILT-IN B-SPLINE FUNCTION, WHICH DOES NOT PROVIDE US WITH GRADIENTS
        # for i in range(self.size):
        #     # x[i].shape = batch_size
        #     spline = BSpline(self.knots, self.coeff[i].detach().numpy(), self.degree)
        #     out[i] = torch.tensor(spline(x[i]))


        # loop through all the spline functions, and apply them to a single element for the whole batch
        # TODO: see if we can vectorize this
        for i in range(self.size):
            # Use torch operations to evaluate the B-spline
            knots = self.knots
            coeff = self.coeff[i]

            # spline_values = self.evaluate_spline(x[i], knots, coeff, self.degree)
            # out[i] = spline_values

            taylor_values = self.evaluate_taylor_series(x[i], coeff, self.degree)
            out[i] = taylor_values

        
        # plot the spline functions (optional)
        # self.plot)


        # reshape the output to be of shape (out_dim, in_dim, batch_size)
        # then we sum it as part of the algorithm
        # then we transpose it so subsequent layers can use it
        y = out.reshape(self.out_dim, self.in_dim, batch_size).sum(dim=1).transpose(0, 1)

        return y
    



    def evaluate_spline(self, x, knots, coeff, degree):
        # Implement the B-spline evaluation directly in PyTorch
        # This is a simplified version and assumes a cubic B-spline (degree=3)
        assert degree == 3, "This implementation only supports cubic B-splines (degree=3)"

        # Initialize the B-spline basis functions
        n_knots = len(knots)
        n_coeffs = len(coeff)
        # assert n_knots == n_coeffs + degree + 1, "Mismatch between number of knots and coefficients for cubic B-splines"

        # Implement basis function evaluation (recursively or using a loop)
        # This example uses a loop for simplicity
        B = torch.zeros(x.shape[0], n_coeffs)

        # Basis function calculation
        for i in range(n_coeffs):
            B[:, i] = self.basis_function(x, knots, i, degree)

        # Evaluate the spline
        spline_values = B.matmul(coeff)

        return spline_values


    def basis_function(self, x, knots, degree):
        # Compute the B-spline basis function value
        # This is a placeholder for the basis function calculation
        pass

        # You need to implement the Cox-de Boor recursion formula here
        


    def evaluate_taylor_series(self, x, coeff, degree):
        exp = torch.arange(degree+1).view(1, -1).repeat(x.shape[0], 1)
        powed = torch.pow(x.view(-1, 1), exp)
        
        return torch.sum(coeff * powed, dim=1)

        



    




    # If we want to plot the spline curves of a layer
    def plot(self):
        # Plot the spline functions (optional)
        points = torch.linspace(self.grid_range[0], self.grid_range[1], 100)

        if self.approx_type == "spline":
            for i in range(self.size):
                spline = BSpline(self.knots.cpu().numpy(), self.coeff[i].detach().cpu().numpy(), self.degree)
                y = spline(points)
                plt.plot(points, y, label=f'B_{i,3}(points)')
        else:
            for i in range(self.size):
                y = self.evaluate_taylor_series(points, coeff=self.coeff[i], degree=self.degree).detach().cpu().numpy()
                plt.plot(points, y, label=f'B_{i,3}(points)')
            
        plt.title('Cubic B-spline Basis Functions')
        plt.xlabel('x')
        plt.ylabel('B_{i,3}(points)')
        plt.legend()
        plt.show()
