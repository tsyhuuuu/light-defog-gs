import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from kan_layer import KANNetwork

from tqdm import tqdm


class AtmosphericKANModel:
    """
    Main KAN model for atmospheric parameter mapping.
    
    This implements a true Kolmogorov-Arnold Network that learns the mapping
    from atmospheric parameters (beta, alpha) to Gaussian splatting parameters
    using learnable univariate functions on edges instead of fixed activations.
    """
    
    def __init__(self, input_dim: int = 7, output_dim: int = 2, 
                 hidden_dims: list = [4, 4], grid_size: int = 5,
                 spline_order: int = 3, noise_scale: float = 0.1,
                 device: str = 'auto'):
        """
        Initialize the KAN model.
        
        Args:
            input_dim: Number of input features (beta, alpha)
            output_dim: Number of output features (Gaussian parameters)
            hidden_dims: List of hidden layer dimensions (smaller for KAN)
            grid_size: Grid resolution for B-spline functions
            spline_order: Order of B-spline polynomials
            noise_scale: Scale for parameter initialization
            device: Device to run the model on
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = KANNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            noise_scale=noise_scale
        ).to(self.device)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        print(f"KAN Model initialized on device: {self.device}")
        self.model.print_architecture()
        
        # Display what makes this KAN special
        print("\nKAN Characteristics:")
        print("✓ Learnable univariate functions on edges (not fixed activations)")
        print("✓ Function decomposition following Kolmogorov-Arnold theorem")
        print("✓ Interpretable learned functions can be visualized")
        print("✓ Efficient representation for smooth multivariate functions")
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = x.to(self.device)
        return self.model(x)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    loss_type: str = 'mse') -> torch.Tensor:
        """Compute loss between predictions and targets."""
        if loss_type == 'mse':
            return nn.MSELoss()(predictions, targets)
        elif loss_type == 'mae':
            return nn.L1Loss()(predictions, targets)
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()(predictions, targets)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def train_epoch(self, train_loader, optimizer, lambda_sparse, loss_type: str = 'mse') -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(x)
            loss = self.compute_loss(predictions, y, loss_type)
            
            # Add KAN-specific sparsity regularization
            if lambda_sparse > 0:
                sparse_loss = self.model.compute_sparsity_loss(lambda_sparse)
                loss = loss + sparse_loss
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader, loss_type: str = 'mse') -> Tuple[float, Dict]:
        """Validate the model for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                predictions = self.model(x)
                loss = self.compute_loss(predictions, y, loss_type)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu())
                all_targets.append(y.cpu())
        
        # Calculate additional metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.calculate_metrics(all_predictions, all_targets)
        avg_loss = total_loss / num_batches
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions: torch.Tensor, 
                         targets: torch.Tensor) -> Dict[str, float]:
        """Calculate various evaluation metrics."""
        mse = nn.MSELoss()(predictions, targets).item()
        mae = nn.L1Loss()(predictions, targets).item()
        
        # R-squared for each output dimension
        r2_scores = []
        for i in range(predictions.shape[1]):
            y_true = targets[:, i].numpy()
            y_pred = predictions[:, i].numpy()
            
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_scores.append(r2)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'r2_min': np.min(r2_scores),
            'r2_max': np.max(r2_scores)
        }
    
    def visualize_learned_functions(self, save_dir: str = "./kan_functions"):
        """
        Visualize the learned univariate functions in the KAN.
        This is a key advantage of KAN - interpretability!
        """
        self.model.visualize_all_functions(save_dir)
        
    def analyze_function_importance(self) -> Dict:
        """
        Analyze which univariate functions are most important.
        This helps understand what the model learned.
        """
        importance_analysis = {}
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer_analysis = {}
            
            for i in range(layer.input_dim):
                for j in range(layer.output_dim):
                    # Measure function complexity by coefficient variance
                    coeffs = layer.phi_functions[i][j].coefficients
                    importance = torch.var(coeffs).item()
                    layer_analysis[f'φ_{i+1},{j+1}'] = importance
            
            importance_analysis[f'layer_{layer_idx}'] = layer_analysis
        
        return importance_analysis
    
    def get_function_representations(self, layer_idx: int = 0, 
                                   num_points: int = 100) -> Dict:
        """
        Extract the learned function representations for analysis.
        
        Args:
            layer_idx: Which layer to analyze
            num_points: Number of points to sample each function
            
        Returns:
            Dictionary with function data for each edge
        """
        if layer_idx >= len(self.model.layers):
            raise ValueError(f"Layer {layer_idx} doesn't exist")
        
        layer = self.model.layers[layer_idx]
        functions_data = {}
        
        for i in range(layer.input_dim):
            for j in range(layer.output_dim):
                func = layer.phi_functions[i][j]
                x_vals, y_vals = func.plot_function(num_points)
                functions_data[f'φ_{i+1},{j+1}'] = {
                    'x_values': x_vals,
                    'y_values': y_vals,
                    'coefficients': func.coefficients.detach().cpu().numpy()
                }
        
        return functions_data

    def train(self, train_loader, val_loader, epochs: int = 100,
              lr: float = 0.001, weight_decay: float = 1e-5,
              loss_type: str = 'mse', patience: int = 20,
              min_delta: float = 1e-6, lambda_sparse: float = 0.0) -> Dict:
        """
        Train the KAN model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            lr: Learning rate
            weight_decay: L2 regularization weight
            loss_type: Type of loss function ('mse', 'mae', 'huber')
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            lambda_sparse: Sparsity regularization weight for KAN
            
        Returns:
            Dictionary with training history and best metrics
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                     lr=lr, weight_decay=weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=patience//2, min_lr=lr*1e-3
        )
        
        best_val_loss = float('inf')
        best_metrics = None
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
        print(f"Loss type: {loss_type}, Patience: {patience}")
        
        for epoch in tqdm(range(epochs)):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, lambda_sparse, loss_type)
            
            # Validate
            val_loss, metrics = self.validate_epoch(val_loader, loss_type)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_metrics = metrics.copy()
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 'best_kan_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  R² Mean: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
                print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_kan_model.pth'))
        
        training_results = {
            'best_val_loss': best_val_loss,
            'best_metrics': best_metrics,
            'total_epochs': epoch + 1,
            'training_history': self.training_history
        }
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best R² score: {best_metrics['r2_mean']:.4f} ± {best_metrics['r2_std']:.4f}")
        
        return training_results
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the model."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x).cpu()
    
    def save_model(self, filepath: str):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'training_history': self.training_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.training_history = checkpoint.get('training_history', {'train_loss': [], 'val_loss': []})
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        plt.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        
        plt.title('KAN Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()