import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import pickle


class AtmosphericDataPreprocessor:
    """Data preprocessing for atmospheric scattering parameter mapping."""
    
    def __init__(self):
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.is_fitted = False
        
        # Define column groups
        self.input_columns = [
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'scale_x', 'scale_y', 'scale_z',
            'opacity'
        ]
        self.output_columns = ['beta', 'alpha']
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(csv_path)
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract input and output features from dataframe."""
        # Input features: beta and alpha
        X = df[self.input_columns].values
        
        # Output features: Gaussian parameters
        y = df[self.output_columns].values
        
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Input range - Beta: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
        print(f"Input range - Alpha: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
        
        return X, y
    
    def fit_scalers(self, X: np.ndarray, y: np.ndarray):
        """Fit the scalers on training data."""
        self.input_scaler.fit(X)
        self.output_scaler.fit(y)
        self.is_fitted = True
        
        print("Scalers fitted successfully")
        print(f"Input scaler mean: {self.input_scaler.mean_}")
        print(f"Input scaler scale: {self.input_scaler.scale_}")
    
    def transform_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before transforming data")
        
        X_scaled = self.input_scaler.transform(X)
        y_scaled = self.output_scaler.transform(y) if y is not None else None
        
        return X_scaled, y_scaled
    
    def inverse_transform_output(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform output predictions."""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before inverse transforming")
        
        return self.output_scaler.inverse_transform(y_scaled)
    
    def create_datasets(self, csv_path: str, test_size: float = 0.2, 
                       validation_size: float = 0.1, random_state: int = 42) -> Dict:
        """Create train/validation/test datasets."""
        # Load data
        df = self.load_data(csv_path)
        X, y = self.prepare_features(df)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Fit scalers on training data
        self.fit_scalers(X_train, y_train)
        
        # Transform all datasets
        X_train_scaled, y_train_scaled = self.transform_data(X_train, y_train)
        X_val_scaled, y_val_scaled = self.transform_data(X_val, y_val)
        X_test_scaled, y_test_scaled = self.transform_data(X_test, y_test)
        
        # Convert to tensors
        datasets = {
            'train': {
                'X': torch.FloatTensor(X_train_scaled),
                'y': torch.FloatTensor(y_train_scaled),
                'X_raw': torch.FloatTensor(X_train),
                'y_raw': torch.FloatTensor(y_train)
            },
            'val': {
                'X': torch.FloatTensor(X_val_scaled),
                'y': torch.FloatTensor(y_val_scaled),
                'X_raw': torch.FloatTensor(X_val),
                'y_raw': torch.FloatTensor(y_val)
            },
            'test': {
                'X': torch.FloatTensor(X_test_scaled),
                'y': torch.FloatTensor(y_test_scaled),
                'X_raw': torch.FloatTensor(X_test),
                'y_raw': torch.FloatTensor(y_test)
            }
        }
        
        print(f"Dataset sizes:")
        print(f"  Train: {len(datasets['train']['X'])}")
        print(f"  Validation: {len(datasets['val']['X'])}")
        print(f"  Test: {len(datasets['test']['X'])}")
        
        return datasets
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to disk."""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before saving")
        
        scaler_data = {
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'input_columns': self.input_columns,
            'output_columns': self.output_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        print(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers from disk."""
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.input_scaler = scaler_data['input_scaler']
        self.output_scaler = scaler_data['output_scaler']
        self.input_columns = scaler_data['input_columns']
        self.output_columns = scaler_data['output_columns']
        self.is_fitted = True
        
        print(f"Scalers loaded from {filepath}")


def create_data_loaders(datasets: Dict, batch_size: int = 32) -> Dict:
    """Create PyTorch data loaders from datasets."""
    from torch.utils.data import DataLoader, TensorDataset
    
    loaders = {}
    for split_name, data in datasets.items():
        dataset = TensorDataset(data['X'], data['y'])
        shuffle = (split_name == 'train')
        loaders[split_name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    
    return loaders