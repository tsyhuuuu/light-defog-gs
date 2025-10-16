#!/usr/bin/env python3
"""
Training script for KAN-based atmospheric parameter mapping.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import AtmosphericDataPreprocessor, create_data_loaders
from kan_model import AtmosphericKANModel


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train KAN model for atmospheric parameter mapping')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='../data/dataset_truck_beta6_alpha250.csv',
                       help='Path to the CSV data file')
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', 
                       default=[4, 4],
                       help='Hidden layer dimensions (smaller for KAN)')
    parser.add_argument('--grid_size', type=int, default=5,
                       help='Grid size for B-spline functions')
    parser.add_argument('--spline_order', type=int, default=3,
                       help='Order of B-spline functions')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                       help='Noise scale for parameter initialization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--loss_type', type=str, default='mse',
                       choices=['mse', 'mae', 'huber'],
                       help='Loss function type')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--lambda_sparse', type=float, default=0.001,
                       help='Sparsity regularization weight for KAN')
    
    # Data split arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size fraction')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size fraction')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--model_name', type=str, default='kan_atmospheric_model',
                       help='Name for saved model files')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment and directories."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.cuda.manual_seed(args.random_seed)
    
    return output_dir, device


def main():
    """Main training function."""
    args = parse_arguments()
    print("=" * 60)
    print("KAN-based Atmospheric Parameter Mapping")
    print("=" * 60)
    
    # Setup environment
    output_dir, device = setup_environment(args)
    
    # Initialize data preprocessor
    print("\n1. Loading and preprocessing data...")
    preprocessor = AtmosphericDataPreprocessor()
    
    # Check if data file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please check the path and try again.")
        return
    
    # Create datasets
    datasets = preprocessor.create_datasets(
        csv_path=str(data_path),
        test_size=args.test_size,
        validation_size=args.val_size,
        random_state=args.random_seed
    )
    
    # Create data loaders
    data_loaders = create_data_loaders(datasets, batch_size=args.batch_size)
    
    # Save preprocessor
    scaler_path = output_dir / f"{args.model_name}_scalers.pkl"
    preprocessor.save_scalers(str(scaler_path))
    
    # Initialize model
    print(f"\n2. Initializing KAN model...")
    input_dim = datasets['train']['X'].shape[1]
    output_dim = datasets['train']['y'].shape[1]
    
    model = AtmosphericKANModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=args.hidden_dims,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        noise_scale=args.noise_scale,
        device=str(device)
    )
    
    print(f"Model architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Hidden dimensions: {args.hidden_dims}")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    # Train model
    print(f"\n3. Training model...")
    training_results = model.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        patience=args.patience,
        lambda_sparse=args.lambda_sparse
    )
    
    # Test model
    print(f"\n4. Evaluating on test set...")
    test_loss, test_metrics = model.validate_epoch(
        data_loaders['test'], loss_type=args.loss_type
    )
    
    print(f"Test Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"  Test MAE: {test_metrics['mae']:.6f}")
    print(f"  Test R² Mean: {test_metrics['r2_mean']:.4f} ± {test_metrics['r2_std']:.4f}")
    print(f"  Test R² Range: [{test_metrics['r2_min']:.4f}, {test_metrics['r2_max']:.4f}]")
    
    # Save model and results
    print(f"\n5. Saving results...")
    model_path = output_dir / f"{args.model_name}.pth"
    model.save_model(str(model_path))
    
    # Save training plot
    plot_path = output_dir / f"{args.model_name}_training_history.png"
    model.plot_training_history(str(plot_path))
    
    # Visualize learned functions (key KAN feature!)
    functions_dir = output_dir / "learned_functions"
    model.visualize_learned_functions(str(functions_dir))
    
    # Analyze function importance
    importance = model.analyze_function_importance()
    import json
    importance_path = output_dir / f"{args.model_name}_function_importance.json"
    with open(importance_path, 'w') as f:
        json.dump(importance, f, indent=2)
    
    print(f"Function importance analysis saved to: {importance_path}")
    
    # Save detailed results
    results = {
        'args': vars(args),
        'training_results': training_results,
        'test_metrics': test_metrics,
        'test_loss': test_loss,
        'model_parameters': model.count_parameters()
    }
    
    import json
    results_path = output_dir / f"{args.model_name}_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"  Model: {model_path}")
    print(f"  Scalers: {scaler_path}")
    print(f"  Results: {results_path}")
    print(f"  Plot: {plot_path}")
    
    # Performance summary
    print(f"\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Final validation R²: {training_results['best_metrics']['r2_mean']:.4f}")
    print(f"Final test R²: {test_metrics['r2_mean']:.4f}")
    print(f"Training epochs: {training_results['total_epochs']}")
    print(f"Model parameters: {model.count_parameters():,}")


if __name__ == "__main__":
    main()