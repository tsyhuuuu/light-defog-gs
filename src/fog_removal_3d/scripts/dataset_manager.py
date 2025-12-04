#!/usr/bin/env python3
"""
Dataset Manager for Multiple CSV Files
Handles automatic discovery, validation, and management of multiple fog Gaussian datasets
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
import logging

from tqdm import tqdm
from datetime import datetime

class DatasetManager:
    """Manages multiple CSV datasets for fog Gaussian classification."""
    
    def __init__(self, data_directory: str, cache_dir: str = "cache"):
        self.data_directory = Path(data_directory)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.datasets = {}
        self.metadata = {}
        self.feature_columns = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False 
        
    def discover_datasets(self, pattern: str = "*.csv", recursive: bool = True) -> List[Path]:
        """Discover all CSV files in the data directory."""
        if recursive:
            csv_files = list(self.data_directory.rglob(pattern))
        else:
            csv_files = list(self.data_directory.glob(pattern))
        
        self.logger.info(f"Discovered {len(csv_files)} CSV files in {self.data_directory}")
        return csv_files
    
    def validate_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single dataset file."""
        validation_info = {
            'file_path': str(file_path),
            'valid': False,
            'error': None,
            'shape': None,
            'columns': None,
            'has_target': False,
            'class_distribution': None,
            'missing_values': None,
            'file_size_mb': None
        }
        
        try:
            # Check file size
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            validation_info['file_size_mb'] = round(file_size, 2)
            
            # Load dataset with memory optimization
            chunk_size = 10000 if file_size > 100 else None
            
            if chunk_size:
                # For large files, sample for validation
                df_sample = pd.read_csv(file_path, nrows=chunk_size)
                df_full_shape = self._get_file_shape(file_path)
                validation_info['shape'] = df_full_shape
            else:
                df_sample = pd.read_csv(file_path)
                validation_info['shape'] = df_sample.shape
            
            validation_info['columns'] = df_sample.columns.tolist()
            
            # Check for required columns
            required_columns = ['is_fog']  # Target column
            essential_columns = ['pos_x', 'pos_y', 'pos_z', 'opacity', 'beta', 'alpha']
            
            has_target = 'is_fog' in df_sample.columns
            has_essential = all(col in df_sample.columns for col in essential_columns)
            
            validation_info['has_target'] = has_target
            validation_info['has_essential'] = has_essential
            
            if has_target:
                # Check class distribution
                if chunk_size:
                    # For large files, estimate distribution
                    class_counts = df_sample['is_fog'].value_counts()
                else:
                    class_counts = df_sample['is_fog'].value_counts()
                
                validation_info['class_distribution'] = class_counts.to_dict()
            
            # Check missing values
            missing_info = df_sample.isnull().sum()
            validation_info['missing_values'] = missing_info[missing_info > 0].to_dict()
            
            # Extract parameters from filename
            params = self._extract_parameters(file_path.name)
            validation_info.update(params)
            
            validation_info['valid'] = has_target and has_essential
            
        except Exception as e:
            validation_info['error'] = str(e)
            self.logger.error(f"Error validating {file_path}: {e}")
        
        return validation_info
    
    def _get_file_shape(self, file_path: Path) -> Tuple[int, int]:
        """Get shape of CSV file without loading it fully."""
        try:
            # Count lines
            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # Subtract header
            
            # Get column count from first row
            df_header = pd.read_csv(file_path, nrows=0)
            col_count = len(df_header.columns)
            
            return (line_count, col_count)
        except:
            return (0, 0)
    
    def _extract_parameters(self, filename: str) -> Dict[str, Any]:
        """Extract beta and alpha parameters from filename."""
        params = {'beta': None, 'alpha': None, 'dataset_type': 'unknown'}
        
        # Common patterns for parameter extraction
        patterns = [
            r'beta(\d+)_alpha(\d+)',
            r'beta(\d+)alpha(\d+)',
            r'density(\d+)',  # Old format
            r'truck.*beta(\d+).*alpha(\d+)'
        ]
        
        filename_lower = filename.lower()
        
        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                if 'density' in pattern:
                    # Convert density to beta (legacy format)
                    density = int(match.group(1))
                    params['beta'] = density / 100.0  # Assuming density is in percentage
                    params['alpha'] = 0  # Default alpha for density format
                else:
                    params['beta'] = int(match.group(1))
                    params['alpha'] = int(match.group(2))
                break
        
        # Determine dataset type
        if 'truck' in filename_lower:
            params['dataset_type'] = 'truck'
        elif 'train' in filename_lower:
            params['dataset_type'] = 'train'
        elif 'test' in filename_lower:
            params['dataset_type'] = 'test'
        elif 'kpro' in filename_lower:
            params['dataset_type'] = 'kpro'
        
        return params
    
    def load_and_validate_all(self) -> Dict[str, Dict]:
        """Load and validate all discovered datasets."""
        csv_files = self.discover_datasets()
        
        validation_results = {}
        valid_count = 0
        
        self.logger.info("Validating datasets...")
        
        for file_path in tqdm(csv_files):
            # self.logger.info(f"Validating: {file_path.name}")
            validation_info = self.validate_dataset(file_path)
            
            dataset_name = file_path.stem
            validation_results[dataset_name] = validation_info
            
            if validation_info['valid']:
                valid_count += 1
        
        self.metadata = validation_results
        
        self.logger.info(f"Validation complete: {valid_count}/{len(csv_files)} datasets are valid")
        
        # Save validation results
        self._save_validation_results(validation_results)
        
        return validation_results
    
    def _save_validation_results(self, results: Dict):
        """Save validation results to cache."""
        cache_file = self.cache_dir / "dataset_validation.json"
        
        # Convert numpy types to native Python types for JSON serialization
        results_serializable = self._make_serializable(results)
        
        with open(cache_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Validation results saved to {cache_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_valid_datasets(self) -> Dict[str, Dict]:
        """Get only valid datasets."""
        if not self.metadata:
            self.load_and_validate_all()
        
        return {name: info for name, info in self.metadata.items() if info['valid']}
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """Get a summary of all datasets."""
        if not self.metadata:
            self.load_and_validate_all()
        
        summary_data = []
        for name, info in self.metadata.items():
            summary_row = {
                'dataset_name': name,
                'valid': info['valid'],
                'rows': info['shape'][0] if info['shape'] else 0,
                'columns': info['shape'][1] if info['shape'] else 0,
                'file_size_mb': info['file_size_mb'],
                'beta': info.get('beta', 'Unknown'),
                'alpha': info.get('alpha', 'Unknown'),
                'dataset_type': info.get('dataset_type', 'Unknown'),
                'fog_count': info['class_distribution'].get(1.0, 0) if info['class_distribution'] else 0,
                'no_fog_count': info['class_distribution'].get(0.0, 0) if info['class_distribution'] else 0,
                'missing_values': len(info['missing_values']) if info['missing_values'] else 0
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data).sort_values(['valid', 'dataset_type', 'beta', 'alpha'], ascending=[False, True, True, True])
    
    def create_dataset_splits(self, strategy: str = 'combined', test_size: float = 0.2, 
                            validation_size: float = 0.1, random_state: int = 42) -> Dict:
        """Create train/validation/test splits using different strategies."""
        valid_datasets = self.get_valid_datasets()
        
        if not valid_datasets:
            raise ValueError("No valid datasets found")
        
        splits = {
            'train': [],
            'validation': [],
            'test': [],
            'strategy': strategy,
            'dataset_info': {}
        }
        
        if strategy == 'combined':
            splits = self._create_combined_splits(valid_datasets, test_size, validation_size, random_state)
        elif strategy == 'cross_dataset':
            splits = self._create_cross_dataset_splits(valid_datasets, test_size, validation_size)
        elif strategy == 'parameter_based':
            splits = self._create_parameter_based_splits(valid_datasets, test_size, validation_size, random_state)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Save split information
        self._save_splits_info(splits)
        
        return splits
    
    def _create_combined_splits(self, datasets: Dict, test_size: float, 
                              validation_size: float, random_state: int) -> Dict:
        """Combine all datasets and split randomly."""
        from sklearn.model_selection import train_test_split
        
        all_data = []
        dataset_sources = []
        
        self.logger.info("Loading and combining all datasets...")
        
        for name, info in tqdm(datasets.items()):
            try:
                df = pd.read_csv(info['file_path'])
                df['source_dataset'] = name
                all_data.append(df)
                dataset_sources.append(name)
                # self.logger.info(f"Loaded {name}: {df.shape}")
            except Exception as e:
                self.logger.error(f"Error loading {name}: {e}")
        
        if not all_data:
            raise ValueError("No datasets could be loaded")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Combined dataset shape: {combined_df.shape}")
        
        # Extract features and target
        feature_columns = self._get_feature_columns(combined_df)
        X = combined_df[feature_columns + ['source_dataset']]
        y = combined_df['is_fog']
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        return {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test),
            'strategy': 'combined',
            'dataset_info': {
                'total_samples': len(combined_df),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'source_datasets': dataset_sources,
                'feature_columns': feature_columns
            }
        }
    
    def _create_cross_dataset_splits(self, datasets: Dict, test_size: float, validation_size: float) -> Dict:
        """Use different datasets for train/validation/test."""
        dataset_list = list(datasets.items())
        
        if len(dataset_list) < 3:
            self.logger.warning("Less than 3 datasets available for cross-dataset split. Using combined strategy.")
            return self._create_combined_splits(datasets, test_size, validation_size, 42)
        
        # Sort by parameters to ensure consistent splitting
        dataset_list.sort(key=lambda x: (x[1].get('beta', 0), x[1].get('alpha', 0)))
        
        n_datasets = len(dataset_list)
        test_count = max(1, int(n_datasets * test_size))
        val_count = max(1, int(n_datasets * validation_size))
        train_count = n_datasets - test_count - val_count
        
        if train_count < 1:
            train_count = 1
            val_count = min(val_count, n_datasets - train_count - test_count)
            test_count = n_datasets - train_count - val_count
        
        train_datasets = dataset_list[:train_count]
        val_datasets = dataset_list[train_count:train_count + val_count]
        test_datasets = dataset_list[train_count + val_count:]
        
        # Load datasets for each split
        splits = {'strategy': 'cross_dataset', 'dataset_info': {}}
        
        for split_name, split_datasets in [('train', train_datasets), ('validation', val_datasets), ('test', test_datasets)]:
            if split_datasets:
                data_list = []
                for name, info in split_datasets:
                    try:
                        df = pd.read_csv(info['file_path'])
                        df['source_dataset'] = name
                        data_list.append(df)
                    except Exception as e:
                        self.logger.error(f"Error loading {name}: {e}")
                
                if data_list:
                    combined_data = pd.concat(data_list, ignore_index=True)
                    feature_columns = self._get_feature_columns(combined_data)
                    X = combined_data[feature_columns + ['source_dataset']]
                    y = combined_data['is_fog']
                    splits[split_name] = (X, y)
                    
                    splits['dataset_info'][f'{split_name}_datasets'] = [name for name, _ in split_datasets]
                    splits['dataset_info'][f'{split_name}_samples'] = len(combined_data)
        
        if 'train' in splits:
            splits['dataset_info']['feature_columns'] = self._get_feature_columns(splits['train'][0])
        
        return splits
    
    def _create_parameter_based_splits(self, datasets: Dict, test_size: float, 
                                     validation_size: float, random_state: int) -> Dict:
        """Split based on beta/alpha parameters."""
        # Group datasets by parameters
        param_groups = {}
        for name, info in datasets.items():
            beta = info.get('beta', 'unknown')
            alpha = info.get('alpha', 'unknown')
            key = f"beta{beta}_alpha{alpha}"
            
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append((name, info))
        
        # Create splits ensuring parameter diversity
        all_combinations = list(param_groups.keys())
        from sklearn.model_selection import train_test_split
        
        if len(all_combinations) >= 3:
            # Split parameter combinations
            train_params, temp_params = train_test_split(
                all_combinations, test_size=(test_size + validation_size), random_state=random_state
            )
            
            if len(temp_params) >= 2:
                val_params, test_params = train_test_split(
                    temp_params, test_size=(test_size / (test_size + validation_size)), random_state=random_state
                )
            else:
                val_params = temp_params[:1] if temp_params else []
                test_params = temp_params[1:] if len(temp_params) > 1 else temp_params
        else:
            # Fall back to combined strategy if not enough parameter combinations
            return self._create_combined_splits(datasets, test_size, validation_size, random_state)
        
        # Load data for each split
        splits = {'strategy': 'parameter_based', 'dataset_info': {}}
        
        for split_name, param_list in [('train', train_params), ('validation', val_params), ('test', test_params)]:
            data_list = []
            dataset_names = []
            
            for param_key in param_list:
                for name, info in param_groups[param_key]:
                    try:
                        df = pd.read_csv(info['file_path'])
                        df['source_dataset'] = name
                        data_list.append(df)
                        dataset_names.append(name)
                    except Exception as e:
                        self.logger.error(f"Error loading {name}: {e}")
            
            if data_list:
                combined_data = pd.concat(data_list, ignore_index=True)
                feature_columns = self._get_feature_columns(combined_data)
                X = combined_data[feature_columns + ['source_dataset']]
                y = combined_data['is_fog']
                splits[split_name] = (X, y)
                
                splits['dataset_info'][f'{split_name}_datasets'] = dataset_names
                splits['dataset_info'][f'{split_name}_samples'] = len(combined_data)
                splits['dataset_info'][f'{split_name}_parameters'] = param_list
        
        if 'train' in splits:
            splits['dataset_info']['feature_columns'] = self._get_feature_columns(splits['train'][0])
        
        return splits
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for model training."""
        if self.feature_columns is not None:
            return self.feature_columns
        
        # Default exclusion if no feature selection config is provided
        exclude_columns = ['beta', 'alpha', 'is_fog', 'pos_x', 'pos_y', 'pos_z', 
                          'rot_0', 'rot_1', 'rot_2', 'rot_3', 'id', 'source_dataset']
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        self.feature_columns = feature_columns
        
        return feature_columns
    
    def set_feature_columns(self, feature_columns: List[str]):
        """Set feature columns externally (e.g., from feature selector)."""
        self.feature_columns = feature_columns
        self.logger.info(f"Feature columns set externally: {len(feature_columns)} features")
    
    def _save_splits_info(self, splits: Dict):
        """Save split information to cache."""
        cache_file = self.cache_dir / "dataset_splits.json"
        
        # Save only the metadata, not the actual data
        splits_info = {
            'strategy': splits['strategy'],
            'dataset_info': splits['dataset_info'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(splits_info, f, indent=2)
        
        self.logger.info(f"Split information saved to {cache_file}")
    
    def print_summary(self):
        """Print a comprehensive summary of all datasets."""
        summary_df = self.get_dataset_summary()
        
        print("\n" + "="*80)
        print("DATASET MANAGER SUMMARY")
        print("="*80)
        
        print(f"Data Directory: {self.data_directory}")
        print(f"Total Datasets: {len(summary_df)}")
        print(f"Valid Datasets: {summary_df['valid'].sum()}")
        print(f"Invalid Datasets: {(~summary_df['valid']).sum()}")
        
        if len(summary_df) > 0:
            total_samples = summary_df['rows'].sum()
            total_size = summary_df['file_size_mb'].sum()
            print(f"Total Samples: {total_samples:,}")
            print(f"Total Size: {total_size:.2f} MB")
            
            print(f"\nDataset Types:")
            type_counts = summary_df['dataset_type'].value_counts()
            for dtype, count in type_counts.items():
                print(f"  {dtype}: {count}")
            
            print(f"\nParameter Distribution:")
            valid_datasets = summary_df[summary_df['valid']]
            if len(valid_datasets) > 0:
                beta_range = f"{valid_datasets['beta'].min()} - {valid_datasets['beta'].max()}"
                alpha_range = f"{valid_datasets['alpha'].min()} - {valid_datasets['alpha'].max()}"
                print(f"  Beta range: {beta_range}")
                print(f"  Alpha range: {alpha_range}")
                
                print(f"\nClass Distribution (Valid Datasets):")
                total_fog = valid_datasets['fog_count'].sum()
                total_no_fog = valid_datasets['no_fog_count'].sum()
                print(f"  Fog samples: {total_fog:,}")
                print(f"  No-fog samples: {total_no_fog:,}")
                if total_fog + total_no_fog > 0:
                    fog_ratio = total_fog / (total_fog + total_no_fog)
                    print(f"  Fog ratio: {fog_ratio:.3f}")
        
        print(f"\nDetailed Dataset Information:")
        print(summary_df.to_string(index=False))
        print("="*130 + "\n")

def main():
    """Example usage of DatasetManager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Manager for Fog Gaussian Classification')
    parser.add_argument('--data-dir', required=True, help='Directory containing CSV datasets')
    parser.add_argument('--cache-dir', default='cache', help='Directory for cache files')
    parser.add_argument('--strategy', choices=['combined', 'cross_dataset', 'parameter_based'], 
                       default='combined', help='Data splitting strategy')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size')
    
    args = parser.parse_args()
    
    # Create dataset manager
    dm = DatasetManager(args.data_dir, args.cache_dir)
    
    # Load and validate datasets
    dm.load_and_validate_all()
    
    # Print summary
    dm.print_summary()
    
    # Create splits
    splits = dm.create_dataset_splits(
        strategy=args.strategy,
        test_size=args.test_size,
        validation_size=args.val_size
    )
    
    print(f"\nDataset splits created using '{args.strategy}' strategy:")
    for split_name in ['train', 'validation', 'test']:
        if split_name in splits and splits[split_name]:
            X, y = splits[split_name]
            print(f"  {split_name}: {len(X)} samples")

if __name__ == "__main__":
    main()