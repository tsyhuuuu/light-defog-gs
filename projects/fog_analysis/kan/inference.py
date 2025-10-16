#!/usr/bin/env python3
"""
Inference script for KAN-based atmospheric parameter mapping.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from itertools import product

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import AtmosphericDataPreprocessor
from kan_model import AtmosphericKANModel


class KANInferenceEngine:
    """Inference engine for trained KAN models."""
    
    def __init__(self, model_path: str, scaler_path: str, device: str = 'auto'):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to saved KAN model
            scaler_path: Path to saved data scalers
            device: Device to run inference on
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load preprocessor with scalers
        self.preprocessor = AtmosphericDataPreprocessor()
        self.preprocessor.load_scalers(scaler_path)
        
        # Load model
        self.model = self._load_model(model_path)
        
        print(f"Inference engine initialized on device: {self.device}")
    
    def _load_model(self, model_path: str) -> AtmosphericKANModel:
        """Load the trained KAN model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        input_dim = checkpoint.get('input_dim', 7)
        output_dim = checkpoint.get('output_dim', 2)
        
        model = AtmosphericKANModel(
            input_dim=input_dim,
            output_dim=output_dim,
            device=str(self.device)
        )
        
        model.load_model(model_path)
        return model
    
    def predict_single(self,
                   f_dc_0: float,
                   f_dc_1: float,
                   f_dc_2: float,
                   scale_x: float,
                   scale_y: float,
                   scale_z: float,
                   opacity: float) -> Dict[str, float]:
        """
        Predict Gaussian parameters for a single input point.

        Args:
            f_dc_0, f_dc_1, f_dc_2: Fourier DC components
            scale_x, scale_y, scale_z: Scaling factors
            opacity: Opacity value

        Returns:
            Dictionary with predicted Gaussian parameters
        """
        # Prepare input
        input_data = np.array([[f_dc_0, f_dc_1, f_dc_2, scale_x, scale_y, scale_z, opacity]])
        input_scaled, _ = self.preprocessor.transform_data(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        # Make prediction
        prediction_scaled = self.model.predict(input_tensor)
        prediction = self.preprocessor.inverse_transform_output(prediction_scaled.numpy())

        # Convert to dictionary
        result = {col: float(prediction[0, i]) for i, col in enumerate(self.preprocessor.output_columns)}
        return result

    
    def predict_batch(self, param_values: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Predict Gaussian parameters for multiple input points.

        Args:
            param_values: Dict of lists containing parameter values for each input.
                        Must contain keys:
                        ['f_dc_0','f_dc_1','f_dc_2','scale_x','scale_y','scale_z','opacity']

        Returns:
            DataFrame with predictions
        """
        # Ensure all parameter lists are the same length
        lengths = [len(v) for v in param_values.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All parameter lists must have the same length")

        # Prepare input
        keys = ['f_dc_0','f_dc_1','f_dc_2','scale_x','scale_y','scale_z','opacity']
        input_data = np.array([[param_values[k][i] for k in keys] for i in range(lengths[0])])
        input_scaled, _ = self.preprocessor.transform_data(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        # Make predictions
        predictions_scaled = self.model.predict(input_tensor)
        predictions = self.preprocessor.inverse_transform_output(predictions_scaled.numpy())

        # Create DataFrame
        result_df = pd.DataFrame(predictions, columns=self.preprocessor.output_columns)
        for k in keys:
            result_df[k] = param_values[k]

        # Reorder columns (inputs first)
        result_df = result_df[keys + self.preprocessor.output_columns]
        return result_df


    def predict_parameter_sweep(self,
                                f_dc_0_range: Tuple[float, float],
                                f_dc_1_range: Tuple[float, float],
                                f_dc_2_range: Tuple[float, float],
                                scale_x_range: Tuple[float, float],
                                scale_y_range: Tuple[float, float],
                                scale_z_range: Tuple[float, float],
                                opacity_range: Tuple[float, float],
                                f_dc_steps: int = 5,
                                scale_steps: int = 5,
                                opacity_steps: int = 5) -> pd.DataFrame:
        """
        Predict Gaussian parameters for a parameter sweep across all 7 inputs.
        """
        # Create value grids
        f_dc_0_values = np.linspace(*f_dc_0_range, f_dc_steps)
        f_dc_1_values = np.linspace(*f_dc_1_range, f_dc_steps)
        f_dc_2_values = np.linspace(*f_dc_2_range, f_dc_steps)
        scale_x_values = np.linspace(*scale_x_range, scale_steps)
        scale_y_values = np.linspace(*scale_y_range, scale_steps)
        scale_z_values = np.linspace(*scale_z_range, scale_steps)
        opacity_values = np.linspace(*opacity_range, opacity_steps)

        # Generate all combinations
        combos = list(product(f_dc_0_values, f_dc_1_values, f_dc_2_values,
                            scale_x_values, scale_y_values, scale_z_values, opacity_values))

        # Build dict of parameter lists
        param_values = {
            'f_dc_0': [c[0] for c in combos],
            'f_dc_1': [c[1] for c in combos],
            'f_dc_2': [c[2] for c in combos],
            'scale_x': [c[3] for c in combos],
            'scale_y': [c[4] for c in combos],
            'scale_z': [c[5] for c in combos],
            'opacity': [c[6] for c in combos],
        }

        return self.predict_batch(param_values)

    
    def visualize_predictions(self, predictions_df: pd.DataFrame, 
                            output_dir: str = "./visualization"):
        """
        Create visualizations of the predictions.

        Args:
            predictions_df: DataFrame with predictions
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Parameter correlation heatmap
        plt.figure(figsize=(12, 10))
        numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = predictions_df[numeric_cols].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                    center=0, fmt='.2f', square=True)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_path / 'parameter_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Input-output relationships
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Pick a few key outputs (example: opacity and scales)
        key_outputs = ['opacity', 'scale_x', 'scale_y', 'scale_z']

        for i, output_param in enumerate(key_outputs[:4]):
            if output_param in predictions_df.columns:
                scatter = axes[i].scatter(predictions_df['f_dc_0'], 
                                        predictions_df[output_param], 
                                        c=predictions_df['f_dc_1'], 
                                        cmap='viridis', alpha=0.6)
                axes[i].set_xlabel('f_dc_0')
                axes[i].set_ylabel(output_param)
                axes[i].set_title(f'{output_param} vs f_dc_0 (colored by f_dc_1)')
                plt.colorbar(scatter, ax=axes[i], label='f_dc_1')

        plt.tight_layout()
        plt.savefig(output_path / 'input_output_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Distribution of predictions
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        output_columns = [col for col in predictions_df.columns 
                        if col not in ['f_dc_0', 'f_dc_1', 'f_dc_2']]

        for i, col in enumerate(output_columns[:len(axes)]):
            axes[i].hist(predictions_df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        # Hide unused subplots
        for i in range(len(output_columns), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(output_path / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_path}")
    
    def export_predictions(self, predictions_df: pd.DataFrame, 
                          output_path: str, format: str = 'csv'):
        """
        Export predictions to file.
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Path to save the file
            format: Export format ('csv', 'json', 'parquet')
        """
        if format.lower() == 'csv':
            predictions_df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            predictions_df.to_json(output_path, orient='records', indent=2)
        elif format.lower() == 'parquet':
            predictions_df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Predictions exported to: {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='KAN model inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--scaler_path', type=str, required=True,
                       help='Path to the saved scalers file')
    
    # Inference mode
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch', 'sweep'],
                       help='Inference mode')
    
    # Single prediction arguments
    parser.add_argument('--f_dc_0', type=float, default=0.0,
                        help='Fourier DC component 0')
    parser.add_argument('--f_dc_1', type=float, default=0.0,
                        help='Fourier DC component 1')
    parser.add_argument('--f_dc_2', type=float, default=0.0,
                        help='Fourier DC component 2')
    parser.add_argument('--scale_x', type=float, default=1.0,
                        help='Scaling factor in X direction')
    parser.add_argument('--scale_y', type=float, default=1.0,
                        help='Scaling factor in Y direction')
    parser.add_argument('--scale_z', type=float, default=1.0,
                        help='Scaling factor in Z direction')
    parser.add_argument('--opacity', type=float, default=1.0,
                        help='Opacity of the object')

    # Parameter sweep arguments
    parser.add_argument('--f_dc_0_min', type=float, default=-1.0,
                        help='Minimum f_dc_0 value for sweep')
    parser.add_argument('--f_dc_0_max', type=float, default=1.0,
                        help='Maximum f_dc_0 value for sweep')
    parser.add_argument('--f_dc_1_min', type=float, default=-1.0,
                        help='Minimum f_dc_1 value for sweep')
    parser.add_argument('--f_dc_1_max', type=float, default=1.0,
                        help='Maximum f_dc_1 value for sweep')
    parser.add_argument('--f_dc_2_min', type=float, default=-1.0,
                        help='Minimum f_dc_2 value for sweep')
    parser.add_argument('--f_dc_2_max', type=float, default=1.0,
                        help='Maximum f_dc_2 value for sweep')

    parser.add_argument('--scale_x_min', type=float, default=0.1,
                        help='Minimum scale_x value for sweep')
    parser.add_argument('--scale_x_max', type=float, default=10.0,
                        help='Maximum scale_x value for sweep')
    parser.add_argument('--scale_y_min', type=float, default=0.1,
                        help='Minimum scale_y value for sweep')
    parser.add_argument('--scale_y_max', type=float, default=10.0,
                        help='Maximum scale_y value for sweep')
    parser.add_argument('--scale_z_min', type=float, default=0.1,
                        help='Minimum scale_z value for sweep')
    parser.add_argument('--scale_z_max', type=float, default=10.0,
                        help='Maximum scale_z value for sweep')

    parser.add_argument('--opacity_min', type=float, default=0.0,
                        help='Minimum opacity value for sweep')
    parser.add_argument('--opacity_max', type=float, default=1.0,
                        help='Maximum opacity value for sweep')

    # Step arguments
    parser.add_argument('--f_dc_steps', type=int, default=10,
                        help='Number of steps for f_dc values in sweep')
    parser.add_argument('--scale_steps', type=int, default=10,
                        help='Number of steps for scale values in sweep')
    parser.add_argument('--opacity_steps', type=int, default=10,
                        help='Number of steps for opacity values in sweep')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--export_format', type=str, default='csv',
                       choices=['csv', 'json', 'parquet'],
                       help='Export format for results')
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("KAN Model Inference")
    print("=" * 60)
    
    # Check if model and scaler files exist
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    if not Path(args.scaler_path).exists():
        print(f"Error: Scaler file not found at {args.scaler_path}")
        return
    
    # Initialize inference engine
    engine = KANInferenceEngine(args.model_path, args.scaler_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform inference based on mode
    if args.mode == 'single':
        print(
            f"\nPredicting for single point: "
            f"f_dc_0={args.f_dc_0}, f_dc_1={args.f_dc_1}, f_dc_2={args.f_dc_2}, "
            f"scale_x={args.scale_x}, scale_y={args.scale_y}, scale_z={args.scale_z}, "
            f"opacity={args.opacity}"
        )

        result = engine.predict_single(
            args.f_dc_0,
            args.f_dc_1,
            args.f_dc_2,
            args.scale_x,
            args.scale_y,
            args.scale_z,
            args.opacity
        )

        print("\nPredicted Gaussian Parameters:")
        print("-" * 40)
        for param, value in result.items():
            print(f"{param:12}: {value:10.6f}")

        # Save single prediction
        result_df = pd.DataFrame([result])

        # Add input args to the dataframe
        result_df['f_dc_0'] = args.f_dc_0
        result_df['f_dc_1'] = args.f_dc_1
        result_df['f_dc_2'] = args.f_dc_2
        result_df['scale_x'] = args.scale_x
        result_df['scale_y'] = args.scale_y
        result_df['scale_z'] = args.scale_z
        result_df['opacity'] = args.opacity

        # Reorder columns (inputs first, then predicted outputs)
        cols = ['f_dc_0', 'f_dc_1', 'f_dc_2',
                'scale_x', 'scale_y', 'scale_z', 'opacity'] + \
            [col for col in result_df.columns
                if col not in ['f_dc_0', 'f_dc_1', 'f_dc_2',
                            'scale_x', 'scale_y', 'scale_z', 'opacity']]
        result_df = result_df[cols]

        output_path = output_dir / f"single_prediction.{args.export_format}"

        engine.export_predictions(result_df, str(output_path), args.export_format)
        
    elif args.mode == 'sweep':
        print(f"\nPerforming parameter sweep:")
        print(f"  f_dc_0 range: [{args.f_dc_0_min}, {args.f_dc_0_max}] ({args.f_dc_steps} steps)")
        print(f"  f_dc_1 range: [{args.f_dc_1_min}, {args.f_dc_1_max}] ({args.f_dc_steps} steps)")
        print(f"  f_dc_2 range: [{args.f_dc_2_min}, {args.f_dc_2_max}] ({args.f_dc_steps} steps)")
        print(f"  scale_x range: [{args.scale_x_min}, {args.scale_x_max}] ({args.scale_steps} steps)")
        print(f"  scale_y range: [{args.scale_y_min}, {args.scale_y_max}] ({args.scale_steps} steps)")
        print(f"  scale_z range: [{args.scale_z_min}, {args.scale_z_max}] ({args.scale_steps} steps)")
        print(f"  opacity range: [{args.opacity_min}, {args.opacity_max}] ({args.opacity_steps} steps)")

        predictions_df = engine.predict_parameter_sweep(
            f_dc_0_range=(args.f_dc_0_min, args.f_dc_0_max),
            f_dc_1_range=(args.f_dc_1_min, args.f_dc_1_max),
            f_dc_2_range=(args.f_dc_2_min, args.f_dc_2_max),
            scale_x_range=(args.scale_x_min, args.scale_x_max),
            scale_y_range=(args.scale_y_min, args.scale_y_max),
            scale_z_range=(args.scale_z_min, args.scale_z_max),
            opacity_range=(args.opacity_min, args.opacity_max),
            f_dc_steps=args.f_dc_steps,
            scale_steps=args.scale_steps,
            opacity_steps=args.opacity_steps
        )

        print(f"\nGenerated {len(predictions_df)} predictions")
        print("\nPrediction statistics:")
        print(predictions_df.describe())

        # Export predictions
        output_path = output_dir / f"parameter_sweep.{args.export_format}"
        engine.export_predictions(predictions_df, str(output_path), args.export_format)

        # Create visualizations if requested
        if args.visualize:
            engine.visualize_predictions(predictions_df, str(output_dir))

    
    print(f"\nInference completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()