import torch
import torch.nn as nn
import spline
import numpy as np
import torch.nn.functional as F

# Main implementation of CKANLayer
class CKANLayer(nn.Module):
    '''
    CKANLayer - A convolutional layer that uses a combination of base and spline functions to approximate a univariate function
    '''

    def __init__(self, in_channels, out_channels, kernel_size, grid, stride=1, padding=0, degree=3, grid_range=[-1, 1], device='cuda:0'):
        super(CKANLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.size = size = in_channels * kernel_size ** 2

        self.stride = stride
        self.padding = padding
        self.degree = degree
        self.grid = grid
        self.grid_range = grid_range
        self.device = device
        self.cache = None
        self.grid

        # Initialize knots and coefficients on the right device during creation
        # knots shape - (size, grid + 1)
        knots = torch \
            .linspace(grid_range[0], grid_range[1], steps=grid + 1, device=device) \
            .view(1, -1) \
            .repeat(size, 1)
        self.knots = nn.Parameter(knots, requires_grad=False)

        # coeff shape - (out_channels, size, grid + degree)
        self.coeff = nn.Parameter(0.1 * torch.randn(out_channels, size, grid + degree, device=device), requires_grad=True)
        
        # Initialize the using Xavier method, as specified in paper
        self.base_weights = torch.nn.Parameter(torch.Tensor(out_channels, size), requires_grad=True)
        nn.init.xavier_uniform_(self.base_weights)  # Xavier uniform initialization
        
        # Initialize with ones, as specified in paper
        self.spline_weights = torch.nn.Parameter(torch.ones(out_channels, size), requires_grad=True)



    # def forward(self, x):
    #     N, _, H, W = x.shape
    #     x_padded = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

    #     # Unfold to get all sliding windows - Shape becomes  (N, C*K*K, L) where L is the number of extracted windows
    #     unfolded = F.unfold(x_padded, kernel_size=self.kernel_size, stride=self.stride, padding=0)
    #     # .transpose(0,1).reshape(self.size, -1)

    #     unfolded.transpose(1, 2).reshape(N, -1, self.in_channels, self.kernel_size, self.kernel_size)
    #     # # Prepare unfolded for batch processing in coef2curve - Final shape becomes (C*K*K, N * L)
    #     # unfolded = unfolded.reshape(self.size, -1)

    #     # store the input for later
    #     self.cache = unfolded

    #     # Output tensor initialization
    #     Hp = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
    #     Wp = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
    #     output = torch.zeros((self.out_channels, N, Hp, Wp), device=self.device)

    #     # Loop through each output channel
    #     for c in range(self.out_channels):
    #         # This calculates w_b*b(x) - Output shape - (1, N * L)
    #         base_values = F.linear(F.silu(unfolded).t(), self.base_weights[c])

    #         # This calculates w_s*spline(x) - Output shape - (1, N * L). Instead of summing the spline values as before, we use (C*K*K, 1) dimensional weights
    #         spline_values = F.linear(spline.coef2curve(unfolded, self.knots, self.coeff[c], self.degree, device=self.device).t(), self.spline_weights[c])

    #         res_values = base_values + spline_values 
    #         output[c] = res_values.view(N, Hp, Wp)
        
    #     return output.transpose(0, 1)


    def forward(self, x):
        N, _, H, W = x.shape
        x_padded = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        # Unfold to get all sliding windows - Shape becomes  (N, C*K*K, L) where L is the number of extracted windows
        unfolded = F.unfold(x_padded, kernel_size=self.kernel_size, stride=self.stride, padding=0)

        unfolded = unfolded.transpose(1, 2).reshape(N, -1, self.in_channels, self.kernel_size, self.kernel_size)

        # Prepare unfolded for batch processing in coef2curve - Final shape becomes (C*K*K, N * L)
        unfolded = unfolded.reshape(-1, self.in_channels * self.kernel_size * self.kernel_size).t()  # (batch_size*Hp*Wp, features)


        # Output tensor initialization
        Hp = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        Wp = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.zeros((N, self.out_channels, Hp, Wp), device=self.device)

        # Loop through each output channel
        for c in range(self.out_channels):
            # This calculates w_b*b(x) - Output shape - (1, N * L)
            base_values = F.linear(F.silu(unfolded).t(), self.base_weights[c]).t()
            # This calculates w_s*spline(x) - Output shape - (1, N * L). Instead of summing the spline values as before, we use (C*K*K, 1) dimensional weights
            spline_values = F.linear(spline.coef2curve(unfolded, self.knots, self.coeff[c], self.degree, device=self.device).t(), self.spline_weights[c], device=self.device).t()
            res_values = base_values + spline_values 
            output[:, c, :, :] = res_values.view(N, Hp, Wp)
        
        return output
    

    

    def update_grid(self, new_grid):
        new_knots = torch \
            .linspace(self.grid_range[0], self.grid_range[1], steps=new_grid + 1, device=self.device) \
            .view(1, -1) \
            .repeat(self.size, 1)
        
        new_coeffs_data = torch.zeros(self.out_channels, self.size, new_grid + self.degree, device=self.device)

        for i in range(self.out_channels):
            # x = self.cache if self.cache is not None else torch.zeros(self.size, 1)
            # as recommended, use the grid points to calculate the new coefficients because they are evenely spaced
            x = new_knots
            y = spline.coef2curve(x, self.knots, self.coeff[i], self.degree, device=self.device)

            new_coeffs_data[i] = spline.curve2coef(x, y, new_knots, self.degree, device=self.device)

        # Update the grid points for the spline
        self.grid = new_grid

        self.coeff = nn.Parameter(new_coeffs_data, requires_grad=True)
        self.knots = nn.Parameter(new_knots, requires_grad=False)



    def update_grid_from_samples(self, x):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        '''
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = spline.coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat([grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.coef.data = spline.curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device)