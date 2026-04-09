"""
losses.py contains code to compute standard data objective 
functions for training Neural Operators. 

By default, losses expect arguments y_pred (model predictions) and y (ground truth.)
"""

import math
from typing import List

import torch


# Set fix{x,y,z}_bnd if function is non-periodic in {x,y,z} direction
# Computes central difference for 1D tensors
# x: (*, s) - input tensor where s is the spatial dimension
# h: scalar - grid spacing
# fix_x_bnd: bool - whether to use forward/backward differences at boundaries
def central_diff_1d(x, h, fix_x_bnd=False):
    # Central difference using torch.roll (periodic by default)
    dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h)

    # Fix boundaries with forward/backward differences if non-periodic
    if fix_x_bnd:
        dx[...,0] = (x[...,1] - x[...,0])/h      # Forward difference at left boundary
        dx[...,-1] = (x[...,-1] - x[...,-2])/h   # Backward difference at right boundary
    
    return dx

# Computes central differences for 2D tensors in both spatial dimensions
# x: (*, s1, s2) - input tensor where s1, s2 are spatial dimensions
# h: scalar or list - grid spacing(s)
# fix_x_bnd, fix_y_bnd: bool - whether to use forward/backward differences at boundaries
def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    # Convert scalar spacing to list for both dimensions
    if isinstance(h, float):
        h = [h, h]

    # Central differences in x and y directions
    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[1])

    # Fix x-direction boundaries if non-periodic
    if fix_x_bnd:
        dx[...,0,:] = (x[...,1,:] - x[...,0,:])/h[0]      # Forward difference
        dx[...,-1,:] = (x[...,-1,:] - x[...,-2,:])/h[0]   # Backward difference
    
    # Fix y-direction boundaries if non-periodic
    if fix_y_bnd:
        dy[...,:,0] = (x[...,:,1] - x[...,:,0])/h[1]      # Forward difference
        dy[...,:,-1] = (x[...,:,-1] - x[...,:,-2])/h[1]   # Backward difference
        
    return dx, dy

# Computes central differences for 3D tensors in all spatial dimensions
# x: (*, s1, s2, s3) - input tensor where s1, s2, s3 are spatial dimensions
# h: scalar or list - grid spacing(s)
# fix_x_bnd, fix_y_bnd, fix_z_bnd: bool - whether to use forward/backward differences at boundaries
def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    # Convert scalar spacing to list for all three dimensions
    if isinstance(h, float):
        h = [h, h, h]

    # Central differences in x, y, and z directions
    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    # Fix x-direction boundaries if non-periodic
    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]    # Forward difference
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0] # Backward difference
    
    # Fix y-direction boundaries if non-periodic
    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]    # Forward difference
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1] # Backward difference
    
    # Fix z-direction boundaries if non-periodic
    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]    # Forward difference
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2] # Backward difference
        
    return dx, dy, dz


# Loss function for computing relative and absolute Lp norms
class LpLoss(object):
    """
    Computes Lp loss with configurable reduction dimensions and operations.
    
    Parameters:
    -----------
    d : int
        Number of spatial dimensions
    p : float  
        Order of the Lp norm (default: 2 for L2 norm)
    L : float or list
        Domain size(s) for computing grid spacing (default: 2*pi)
    reduce_dims : int or list
        Dimensions over which to reduce (default: 0)
    reductions : str or list
        Type of reduction ('sum' or 'mean') for each reduce_dim
    """
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        super().__init__()

        self.d = d  # Number of spatial dimensions
        self.p = p  # Lp norm order

        # Convert reduce_dims to list format
        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        # Set up reduction operations for each dimension
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        # Convert domain size to list format
        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def uniform_h(self, x):
        """Compute uniform grid spacing based on tensor size and domain length"""
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        """Apply all specified reductions to the tensor"""
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y, h=None):
        """Compute absolute Lp loss between x and y"""
        # Assume uniform mesh - compute grid spacing if not provided
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        # Scale by grid volume factor
        const = math.prod(h)**(1.0/self.p)
        # Compute Lp norm of flattened difference
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        # Apply reductions if specified
        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        """Compute relative Lp loss between x and y"""
        # Compute Lp norm of difference
        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        # Compute Lp norm of ground truth for normalization
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        # Normalize difference by ground truth norm
        diff = diff/ynorm

        # Apply reductions if specified
        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        """Default forward pass computes relative loss"""
        return self.rel(y_pred, y)


class H1Loss(object):
    """
    H1 loss function that includes both function values and first derivatives.
    Useful for problems where smoothness of the solution is important.
    
    Parameters:
    -----------
    d : int
        Number of spatial dimensions (1, 2, or 3)
    L : float or list
        Domain size(s) for computing grid spacing
    reduce_dims : int or list
        Dimensions over which to reduce
    reductions : str or list
        Type of reduction ('sum' or 'mean') for each reduce_dim
    fix_x_bnd, fix_y_bnd, fix_z_bnd : bool
        Whether to use non-periodic boundary conditions in each direction
    """
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd

        # Convert reduce_dims to list format
        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        # Set up reduction operations
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        # Convert domain size to list format
        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def compute_terms(self, x, y, h):
        """Compute function values and derivatives for H1 norm calculation"""
        dict_x = {}  # Store function and derivative terms for x
        dict_y = {}  # Store function and derivative terms for y

        if self.d == 1:
            # Store function values
            dict_x[0] = x
            dict_y[0] = y

            # Compute first derivatives
            x_x = central_diff_1d(x, h[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, h[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x
        
        elif self.d == 2:
            # Flatten function values for 2D case
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            # Compute partial derivatives in x and y directions
            x_x, x_y = central_diff_2d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            # Flatten derivatives
            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)
        
        else:  # d == 3
            # Flatten function values for 3D case
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            # Compute partial derivatives in x, y, and z directions
            x_x, x_y, x_z = central_diff_3d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            # Flatten derivatives
            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)
        
        return dict_x, dict_y

    def uniform_h(self, x):
        """Compute uniform grid spacing based on tensor size and domain length"""
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h
    
    def reduce_all(self, x):
        """Apply all specified reductions to the tensor"""
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x
        
    def abs(self, x, y, h=None):
        """Compute absolute H1 loss (includes function values and derivatives)"""
        # Assume uniform mesh - compute grid spacing if not provided
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
            
        # Get function values and derivatives
        dict_x, dict_y = self.compute_terms(x, y, h)

        # Scale by grid volume and compute L2 norm of function difference
        const = math.prod(h)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        # Add L2 norms of derivative differences
        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        # Take square root to get H1 norm
        diff = diff**0.5

        # Apply reductions if specified
        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, h=None):
        """Compute relative H1 loss"""
        # Assume uniform mesh - compute grid spacing if not provided
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        # Get function values and derivatives
        dict_x, dict_y = self.compute_terms(x, y, h)

        # Compute squared L2 norms of differences and ground truth
        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        # Add contributions from derivatives
        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        # Normalize difference by ground truth H1 norm
        diff = (diff**0.5)/(ynorm**0.5)

        # Apply reductions if specified
        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, h=None, **kwargs):
        """Default forward pass computes relative H1 loss"""
        return self.rel(y_pred, y, h=h)


class IregularLpqLoss(torch.nn.Module):
    """
    Lp,q loss for irregular grids with volume elements.
    
    Parameters:
    -----------
    p : float
        Outer norm order (default: 2.0)
    q : float  
        Inner norm order (default: 2.0)
    """
    def __init__(self, p=2.0, q=2.0):
        super().__init__()

        self.p = 2.0  # Outer norm order
        self.q = 2.0  # Inner norm order
    
    def norm(self, x, vol_elm):
        """
        Compute weighted Lp,q norm using volume elements
        
        Parameters:
        -----------
        x : tensor (n, c) or (n,)
            Input values at grid points
        vol_elm : tensor (n,)
            Volume elements for integration weights
        """
        if len(x.shape) > 1:
            # For multi-channel data, compute q-norm across channels first
            s = torch.sum(torch.abs(x)**self.q, dim=1, keepdim=False)**(self.p/self.q)
        else:
            # For single channel, use p-norm directly
            s = torch.abs(x)**self.p
        
        # Integrate using volume elements and take p-th root
        return torch.sum(s*vol_elm)**(1.0/self.p)

    def abs(self, x, y, vol_elm):
        """Compute absolute weighted loss"""
        return self.norm(x - y, vol_elm)
    
    def rel(self, x, y, vol_elm):
        """Compute relative weighted loss (normalized by ground truth norm)"""
        return self.abs(x, y, vol_elm)/self.norm(y, vol_elm)
    
    def forward(self, y_pred, y, vol_elm, **kwargs):
        """Default forward pass computes relative loss"""
        return self.rel(y_pred, y, vol_elm)


def pressure_drag(pressure, vol_elm, inward_surface_normal, 
                  flow_direction_normal, flow_speed, 
                  reference_area, mass_density=1.0):
    """
    Compute pressure drag coefficient from surface pressure distribution.
    
    Parameters:
    -----------
    pressure : tensor
        Pressure values on surface elements
    vol_elm : tensor
        Surface area elements for integration
    inward_surface_normal : tensor
        Inward pointing surface normal vectors  
    flow_direction_normal : tensor
        Flow direction unit vector
    flow_speed : float
        Reference flow speed
    reference_area : float
        Reference area for normalization
    mass_density : float
        Fluid density (default: 1.0)
    """
    # Drag coefficient normalization constant
    const = 2.0/(mass_density*(flow_speed**2)*reference_area)
    # Project surface normal onto flow direction
    direction = torch.sum(inward_surface_normal*flow_direction_normal, dim=1, keepdim=False)
    
    # Integrate pressure forces in flow direction
    return const*torch.sum(pressure*direction*vol_elm)

def friction_drag(wall_shear_stress, vol_elm, 
                  flow_direction_normal, flow_speed, 
                  reference_area, mass_density=1.0):
    """
    Compute friction drag coefficient from wall shear stress distribution.
    
    Parameters:
    -----------
    wall_shear_stress : tensor
        Wall shear stress vectors on surface elements
    vol_elm : tensor
        Surface area elements for integration
    flow_direction_normal : tensor
        Flow direction unit vector
    flow_speed : float
        Reference flow speed
    reference_area : float
        Reference area for normalization
    mass_density : float
        Fluid density (default: 1.0)
    """
    # Drag coefficient normalization constant
    const = 2.0/(mass_density*(flow_speed**2)*reference_area)
    # Project shear stress onto flow direction
    direction = torch.sum(wall_shear_stress*flow_direction_normal, dim=1, keepdim=False)

    # Note: unused variable x (appears to be debugging artifact)
    x = torch.sum(direction*vol_elm)

    # Integrate viscous forces in flow direction
    return const*torch.sum(direction*vol_elm)

def total_drag(pressure, wall_shear_stress, vol_elm, 
               inward_surface_normal, flow_direction_normal, 
               flow_speed, reference_area, mass_density=1.0):
    """
    Compute total drag coefficient as sum of pressure and friction components.
    
    Parameters match those of pressure_drag and friction_drag functions.
    """
    # Compute pressure drag component
    cp = pressure_drag(pressure, vol_elm, inward_surface_normal, 
                       flow_direction_normal, flow_speed, 
                       reference_area, mass_density)
    
    # Compute friction drag component
    cf = friction_drag(wall_shear_stress, vol_elm, 
                       flow_direction_normal, flow_speed, 
                       reference_area, mass_density)
    
    # Return total drag coefficient
    return cp + cf 


class WeightedL2DragLoss(object):
    """
    Computes relative L2 loss between predicted and true drag coefficients.
    Useful for training models to predict aerodynamic quantities.
    
    Parameters:
    -----------
    mappings : dict
        Dictionary mapping field names to tensor indices for extracting
        'wall_shear_stress' and 'pressure' from model outputs
    device : str
        Device for tensor computations (default: 'cuda')
    """

    def __init__(self, mappings: dict, device: str = 'cuda'):
        """Initialize drag loss with field mappings and device."""
        super().__init__()
        self.mappings = mappings  # Maps field names to tensor slice indices
        self.device = device


    def __call__(self, y_pred, y, vol_elm, inward_normals, flow_normals, flow_speed, reference_area, **kwargs):
        """
        Compute relative drag coefficient loss.
        
        Parameters:
        -----------
        y_pred : tensor
            Model predictions containing pressure and shear stress
        y : tensor
            Ground truth values  
        vol_elm : tensor
            Surface area elements
        inward_normals : tensor
            Inward surface normal vectors
        flow_normals : tensor
            Flow direction vectors
        flow_speed : float
            Reference flow speed
        reference_area : float
            Reference area for drag coefficient
        """
        c_pred = None
        c_truth = None
        loss = 0.
        
        # Extract wall shear stress from predictions and ground truth
        stress_indices = self.mappings['wall_shear_stress']
        pred_stress = y_pred[stress_indices].view(-1,1)
        truth_stress = y[stress_indices]

        # Pad stress to 3D vectors (friction drag function expects 3 components)
        pred_stress_pad = torch.zeros((pred_stress.shape[0], 3), device=self.device)
        pred_stress_pad[:,0] = pred_stress.view(-1,)

        truth_stress_pad = torch.zeros((truth_stress.shape[0], 3), device=self.device)
        truth_stress_pad[:,0] = truth_stress.view(-1,)

        # Extract pressure from predictions and ground truth
        pressure_indices = self.mappings['pressure']
        pred_pressure = y_pred[pressure_indices].view(-1,1)
        truth_pressure = y[pressure_indices]

        # Compute predicted total drag coefficient
        c_pred = total_drag(pressure=pred_pressure,
                            wall_shear_stress=pred_stress_pad,
                            vol_elm=vol_elm,
                            inward_surface_normal=inward_normals,
                            flow_direction_normal=flow_normals,
                            flow_speed=flow_speed,
                            reference_area=reference_area
                            )
        # Compute ground truth total drag coefficient
        c_truth = total_drag(pressure=truth_pressure,
                            wall_shear_stress=truth_stress_pad,
                            vol_elm=vol_elm,
                            inward_surface_normal=inward_normals,
                            flow_direction_normal=flow_normals,
                            flow_speed=flow_speed,
                            reference_area=reference_area
                            )

        # Compute relative error in drag coefficient
        loss += torch.abs(c_pred - c_truth) / torch.abs(c_truth)

        # Scale loss by number of fields plus one
        loss = (1.0/len(self.mappings) + 1)*loss

        return loss


def variance_aware_multicol_loss(
    y_pred,
    y_true,
    weights,
    output_window_size,
    target_cols,
    lambda_conc_focus=0.5,
):
    """
    Compute variance-aware multi-column loss using pre-computed weights from the dataset.
    
    This loss function combines a global loss over all variables with a concentration-specific
    variance-weighted loss. It's designed for multi-variable time series prediction where
    different variables (e.g., concentration, head, pressure) are concatenated along the
    last dimension.
    
    Args:
        y_pred (torch.Tensor): Predicted values [B, N_points, T_out * C]
        y_true (torch.Tensor): Target values [B, N_points, T_out * C]
        weights (torch.Tensor): Pre-computed variance-aware weights [N_points]
        output_window_size (int): T_out (number of output timesteps)
        target_cols (list): List of target column names like ['mass_concentration', 'head']
        lambda_conc_focus (float): Weight factor for concentration-focused loss (0 to 1)
    
    Returns:
        tuple: (total_loss, global_loss, conc_var_loss)
            - total_loss: Combined loss value
            - global_loss: Global MSE over all variables (detached)
            - conc_var_loss: Variance-weighted concentration loss (detached)
    
    Note:
        This function assumes 'mass_concentration' is one of the target columns.
        The weights should be pre-computed based on temporal variances across the dataset
        and normalized to have mean 1.0.
    """

    B, N, TC = y_pred.shape
    C = TC // output_window_size
    assert TC == output_window_size * C, f"Shape mismatch: {TC} != {output_window_size} * {C}"

    # reshape to [B, N, T_out, C]
    y_pred = y_pred.view(B, N, output_window_size, C)
    y_true = y_true.view(B, N, output_window_size, C)

    # Global loss over all variables
    global_loss_fn = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions='mean')
    global_loss = global_loss_fn(y_pred, y_true)

    # Variance-aware term: MSE for concentration (absolute error avoids division-by-zero)
    conc_idx = target_cols.index('mass_concentration')
    conc_pred = y_pred[..., conc_idx]   # [B, N, T]
    conc_true = y_true[..., conc_idx]   # [B, N, T]

    weights = weights.to(y_pred.device)
    mse_per_node = ((conc_pred - conc_true) ** 2).mean(dim=[0, 2])  # [N]
    weighted_mse = (weights * mse_per_node).mean()
    conc_var_loss = torch.sqrt(weighted_mse + 1e-8) 

    # Combine losses
    loss = global_loss + lambda_conc_focus * conc_var_loss

    return loss, global_loss.detach(), conc_var_loss.detach()