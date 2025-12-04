import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class BSplineActivation(nn.Module):
    """
    Learnable B-spline activation function for KAN edges.
    This represents a univariate function φ(x) that can be learned.
    """
    
    def __init__(self, grid_size: int = 5, spline_order: int = 3, 
                 input_range: tuple = (-1, 1), noise_scale: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.input_range = input_range
        
        # Create extended grid for B-splines
        grid_extended = torch.linspace(
            input_range[0] - 0.1 * (input_range[1] - input_range[0]),
            input_range[1] + 0.1 * (input_range[1] - input_range[0]),
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid_extended)
        
        # Learnable spline coefficients - this is where the nonlinear function is represented
        self.coefficients = nn.Parameter(torch.randn(grid_size + spline_order) * noise_scale)
        
        # Initialize with small random values
        nn.init.normal_(self.coefficients, 0, noise_scale)
    
    def b_spline_basis(self, x: torch.Tensor, i: int, k: int) -> torch.Tensor:
        """Compute B-spline basis function B_i,k(x) using Cox-de Boor recursion."""
        if k == 0:
            # Base case: indicator function
            return ((self.grid[i] <= x) & (x < self.grid[i + 1])).float()
        
        # Recursive case
        left_term = torch.zeros_like(x)
        right_term = torch.zeros_like(x)
        
        # Left term
        denom1 = self.grid[i + k] - self.grid[i]
        if denom1 > 1e-8:  # Avoid division by zero
            left_term = (x - self.grid[i]) / denom1 * self.b_spline_basis(x, i, k - 1)
        
        # Right term
        denom2 = self.grid[i + k + 1] - self.grid[i + 1]
        if denom2 > 1e-8:  # Avoid division by zero
            right_term = (self.grid[i + k + 1] - x) / denom2 * self.b_spline_basis(x, i + 1, k - 1)
        
        return left_term + right_term
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the learned univariate function φ(x).
        This is the core of KAN - each edge has its own learnable function.
        """
        # Clamp input to reasonable range
        x_clamped = torch.clamp(x, self.input_range[0], self.input_range[1])
        
        # Compute spline function as linear combination of basis functions
        output = torch.zeros_like(x_clamped)
        
        for i in range(len(self.coefficients)):
            basis_val = self.b_spline_basis(x_clamped, i, self.spline_order)
            output += self.coefficients[i] * basis_val
        
        return output
    
    def plot_function(self, num_points: int = 1000) -> tuple:
        """Plot the learned univariate function for visualization."""
        x_vals = torch.linspace(self.input_range[0], self.input_range[1], num_points)
        with torch.no_grad():
            y_vals = self.forward(x_vals)
        return x_vals.numpy(), y_vals.numpy()


class KANLayer(nn.Module):
    """
    True Kolmogorov-Arnold Network Layer.
    
    In KAN, each edge between nodes has a learnable univariate function φ_ij(x).
    The key insight is that complex multivariate functions can be represented as 
    compositions of univariate functions, following the Kolmogorov-Arnold theorem.
    """
    
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 5, 
                 spline_order: int = 3, noise_scale: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create learnable univariate functions for each edge (i,j)
        # φ_ij: input_i -> contribution to output_j
        self.phi_functions = nn.ModuleList([
            nn.ModuleList([
                BSplineActivation(grid_size, spline_order, noise_scale=noise_scale)
                for _ in range(output_dim)
            ]) for _ in range(input_dim)
        ])
        
        # Optional: Add residual connection for easier optimization
        self.use_residual = True
        if self.use_residual:
            self.residual_linear = nn.Linear(input_dim, output_dim, bias=False)
            # Initialize with small weights
            nn.init.normal_(self.residual_linear.weight, 0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the KAN computation:
        
        y_j = Σ_i φ_ij(x_i)
        
        Each output is a sum of learned univariate functions applied to each input.
        """
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        # Apply univariate functions φ_ij to each input-output pair
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                # φ_ij(x_i) contributes to output y_j
                contribution = self.phi_functions[i][j](x[:, i])
                output[:, j] += contribution
        
        # Add residual connection for easier optimization
        if self.use_residual:
            output += self.residual_linear(x)
        
        return output
    
    def get_edge_function(self, input_idx: int, output_idx: int) -> BSplineActivation:
        """Get the univariate function for a specific edge."""
        return self.phi_functions[input_idx][output_idx]
    
    def visualize_functions(self, save_path: str = None):
        """Visualize all learned univariate functions."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(self.input_dim, self.output_dim, 
                                figsize=(3 * self.output_dim, 3 * self.input_dim))
        if self.input_dim == 1:
            axes = axes.reshape(1, -1)
        if self.output_dim == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                x_vals, y_vals = self.phi_functions[i][j].plot_function()
                axes[i, j].plot(x_vals, y_vals, 'b-', linewidth=2)
                axes[i, j].set_title(f'φ_{i+1},{j+1}(x)')
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_xlabel(f'Input {i+1}')
                axes[i, j].set_ylabel(f'Output contribution to {j+1}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class KANNetwork(nn.Module):
    """
    Complete Kolmogorov-Arnold Network.
    
    This implements the true KAN architecture where complex multivariate functions
    are decomposed into compositions of learnable univariate functions.
    
    The network represents the function f: R^n -> R^m as:
    f(x) = Φ_L ∘ Φ_{L-1} ∘ ... ∘ Φ_1(x)
    
    where each Φ_l is a KAN layer with learnable univariate functions on edges.
    """
    
    def __init__(self, input_dim: int = 7, output_dim: int = 2, 
                 hidden_dims: List[int] = [4, 4], grid_size: int = 5, 
                 spline_order: int = 3, noise_scale: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Input normalization for better training stability
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Build sequence of KAN layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            kan_layer = KANLayer(
                input_dim=dims[i], 
                output_dim=dims[i + 1],
                grid_size=grid_size,
                spline_order=spline_order,
                noise_scale=noise_scale
            )
            self.layers.append(kan_layer)
        
        # Keep track of architecture for interpretability
        self.architecture = dims
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KAN network.
        
        The key difference from MLPs: no fixed activation functions!
        Each layer applies learnable univariate functions on edges.
        """
        # Normalize inputs
        x = self.input_norm(x)
        
        # Pass through KAN layers
        # Each layer applies: y_j = Σ_i φ_ij(x_i)
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_layer_functions(self, layer_idx: int) -> KANLayer:
        """Get the KAN layer at specified index for analysis."""
        return self.layers[layer_idx]
    
    def visualize_layer_functions(self, layer_idx: int, save_path: str = None):
        """Visualize the learned univariate functions in a specific layer."""
        if layer_idx >= len(self.layers):
            raise ValueError(f"Layer index {layer_idx} out of range. Network has {len(self.layers)} layers.")
        
        layer = self.layers[layer_idx]
        layer.visualize_functions(save_path)
    
    def visualize_all_functions(self, save_dir: str = "./kan_functions"):
        """Visualize learned functions in all layers."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for i, layer in enumerate(self.layers):
            save_path = os.path.join(save_dir, f"layer_{i}_functions.png")
            layer.visualize_functions(save_path)
            print(f"Saved layer {i} functions to {save_path}")
    
    def get_function_complexity(self) -> dict:
        """
        Analyze the complexity of learned functions.
        This is a key advantage of KAN - interpretability!
        """
        complexity_info = {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'layer_info': []
        }
        
        for i, layer in enumerate(self.layers):
            layer_info = {
                'layer_idx': i,
                'input_dim': layer.input_dim,
                'output_dim': layer.output_dim,
                'num_functions': layer.input_dim * layer.output_dim,
                'parameters_per_function': len(layer.phi_functions[0][0].coefficients)
            }
            complexity_info['layer_info'].append(layer_info)
        
        return complexity_info
    
    def print_architecture(self):
        """Print the network architecture."""
        print("KAN Architecture:")
        print(f"Input dimension: {self.input_dim}")
        for i, (input_dim, output_dim) in enumerate(zip(self.architecture[:-1], self.architecture[1:])):
            print(f"Layer {i}: {input_dim} -> {output_dim} ({input_dim * output_dim} univariate functions)")
        print(f"Output dimension: {self.output_dim}")
        
        complexity = self.get_function_complexity()
        print(f"Total parameters: {complexity['total_parameters']:,}")
        
    def compute_sparsity_loss(self, lambda_sparse: float = 0.01) -> torch.Tensor:
        """
        Compute sparsity regularization for KAN.
        This encourages the network to use fewer active functions.
        """
        sparse_loss = 0.0
        
        for layer in self.layers:
            for i in range(layer.input_dim):
                for j in range(layer.output_dim):
                    # L1 penalty on function coefficients
                    coeffs = layer.phi_functions[i][j].coefficients
                    sparse_loss += torch.norm(coeffs, p=1)
        
        return lambda_sparse * sparse_loss