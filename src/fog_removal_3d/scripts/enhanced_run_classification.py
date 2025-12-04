#!/usr/bin/env python3
"""
Enhanced Multi-Dataset Fog Gaussian Classification Runner
Supports automatic dataset discovery, multiple splitting strategies, and comprehensive experiment tracking
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import lightgbm as lgb
import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from dataset_manager import DatasetManager
from experiment_tracker import ExperimentTracker
from feature_selector import FeatureSelector

class MultiDatasetClassificationRunner:
    """Enhanced runner for multi-dataset fog Gaussian classification experiments."""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.dataset_manager = DatasetManager(
            self.config['dataset']['data_directory'],
            self.config['dataset']['cache_directory']
        )
        
        self.experiment_tracker = ExperimentTracker(
            self.config['experiment']['name'],
            self.config['experiment']['output_directory']
        )
        
        self.feature_selector = FeatureSelector(
            self.config,
            self.experiment_tracker.logger
        )
        
        # Setup logging
        self.logger = self.experiment_tracker.logger
        
        # Initialize data splits
        self.data_splits = None
        self.scaler = None
    
    def prepare_datasets(self) -> Dict:
        """Discover, validate, and prepare datasets."""
        self.logger.info("=" * 60)
        self.logger.info("DATASET PREPARATION")
        self.logger.info("=" * 60)
        
        # Discover and validate datasets
        self.dataset_manager.load_and_validate_all()
        
        # Print dataset summary
        self.dataset_manager.print_summary()
        
        # Create data splits
        splits_config = self.config['dataset']['splitting']
        self.data_splits = self.dataset_manager.create_dataset_splits(
            strategy=splits_config['strategy'],
            test_size=splits_config['test_size'],
            validation_size=splits_config['validation_size'],
            random_state=splits_config['random_state']
        )
        
        # Log dataset info to experiment tracker
        self.experiment_tracker.log_dataset_info(self.data_splits['dataset_info'])
        
        # Prepare data for training
        return self._prepare_training_data()
    
    def _prepare_training_data(self) -> Dict:
        """Prepare training data with preprocessing and feature selection."""
        prepared_data = {}
        
        # Perform feature selection on the combined training data
        if 'train' in self.data_splits and self.data_splits['train']:
            X_train, y_train = self.data_splits['train']
            
            # Create a sample dataframe for feature selection
            sample_df = X_train.copy()
            sample_df['is_fog'] = y_train
            
            # Select features using the feature selector
            selected_features = self.feature_selector.select_features(sample_df, 'is_fog')
            
            # Update dataset manager with selected features
            self.dataset_manager.set_feature_columns(selected_features)
            
            # Save feature selection info
            feature_info_path = self.experiment_tracker.results_dir / "feature_selection_info.json"
            self.feature_selector.save_feature_info(str(feature_info_path))
            
            # Log feature importance if available
            importance_df = self.feature_selector.get_feature_importance_summary()
            if not importance_df.empty:
                importance_path = self.experiment_tracker.results_dir / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                self.logger.info(f"Feature importance saved to {importance_path}")
            
        else:
            # Fallback to default feature selection
            selected_features = self.data_splits['dataset_info']['feature_columns']
        
        # Prepare data splits with selected features
        for split_name in ['train', 'validation', 'test']:
            if split_name in self.data_splits and self.data_splits[split_name]:
                X, y = self.data_splits[split_name]
                
                # Use selected features (validate they exist)
                available_features = [f for f in selected_features if f in X.columns]
                if len(available_features) < len(selected_features):
                    missing = set(selected_features) - set(available_features)
                    self.logger.warning(f"Missing features in {split_name} split: {missing}")
                
                X_features = X[available_features]
                
                prepared_data[split_name] = {
                    'X': X_features,
                    'y': y,
                    'source_info': X['source_dataset'] if 'source_dataset' in X.columns else None
                }
        
        # Apply preprocessing
        if self.config['dataset']['preprocessing']['normalize_features']:
            prepared_data = self._apply_normalization(prepared_data)
        
        # Update experiment info with feature selection details
        self.experiment_tracker.experiment_info['feature_selection'] = {
            'method': self.config['dataset']['features']['selection_mode'],
            'selected_features': selected_features,
            'num_features': len(selected_features)
        }
        
        return prepared_data
    
    def _apply_normalization(self, data: Dict) -> Dict:
        """Apply feature normalization."""
        self.logger.info("Applying feature normalization...")
        
        self.scaler = StandardScaler()
        
        # Fit on training data
        if 'train' in data:
            X_train_scaled = self.scaler.fit_transform(data['train']['X'])
            data['train']['X'] = pd.DataFrame(X_train_scaled, 
                                            columns=data['train']['X'].columns,
                                            index=data['train']['X'].index)
        
        # Transform validation and test data
        for split_name in ['validation', 'test']:
            if split_name in data:
                X_scaled = self.scaler.transform(data[split_name]['X'])
                data[split_name]['X'] = pd.DataFrame(X_scaled,
                                                   columns=data[split_name]['X'].columns,
                                                   index=data[split_name]['X'].index)
        
        return data
    
    def run_svm_classification(self, data: Dict) -> Dict:
        """Run SVM classification with hyperparameter tuning."""
        if not self.config['models']['svm']['enabled']:
            return {}
        
        self.logger.info("=" * 60)
        self.logger.info("SVM CLASSIFICATION")
        self.logger.info("=" * 60)
        
        self.experiment_tracker.start_model_training("SVM", self.config['models']['svm'])
        
        results = {}
        
        try:
            # Get hyperparameters
            hyperparams = self.config['models']['svm']['hyperparameters']
            
            # Simple grid search for best parameters
            best_score = 0
            best_params = {}
            best_model = None
            
            X_train = data['train']['X'].values
            y_train = data['train']['y'].values
            
            for kernel in hyperparams['kernel']:
                for C in hyperparams['C']:
                    for gamma in hyperparams['gamma']:
                        self.logger.info(f"Current params: kernel={kernel}, C={C}, gamma={gamma}")
                        # Create model
                        model = SVC(kernel=kernel, C=C, gamma=gamma, 
                                  probability=True, random_state=42)
                        
                        # Cross-validation
                        if self.config['experiment']['evaluation']['cross_validate']:
                            cv = StratifiedKFold(n_splits=self.config['dataset']['splitting']['cv_folds'], 
                                               shuffle=True, random_state=42)
                            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                            mean_score = scores.mean()
                        else:
                            model.fit(X_train, y_train)
                            mean_score = model.score(X_train, y_train)
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}
                            if not self.config['experiment']['evaluation']['cross_validate']:
                                best_model = model
            
            self.logger.info(f"Best SVM parameters: {best_params}")
            self.logger.info(f"Best CV score: {best_score:.4f}")
            
            # Train final model with best parameters
            if best_model is None:
                best_model = SVC(**best_params, probability=True, random_state=42)
                best_model.fit(X_train, y_train)
            
            # Evaluate on all splits
            for split_name in ['train', 'validation', 'test']:
                if split_name in data:
                    X = data[split_name]['X'].values
                    y = data[split_name]['y'].values
                    
                    y_pred = best_model.predict(X)
                    y_prob = best_model.predict_proba(X)[:, 1]
                    
                    metrics = self.experiment_tracker.evaluate_model(
                        "SVM", y, y_pred, y_prob, split_name
                    )
                    results[split_name] = metrics
                    
                    # Create visualizations for test set
                    if split_name == 'test':
                        self.experiment_tracker.create_confusion_matrix_plot("SVM", y, y_pred)
                        self.experiment_tracker.create_roc_curve_plot("SVM", y, y_prob)
                        self.experiment_tracker.create_precision_recall_curve_plot("SVM", y, y_prob)
            
            self.experiment_tracker.finish_model_training("SVM", best_model, results.get('test', {}))
            
        except Exception as e:
            self.logger.error(f"SVM classification failed: {e}")
            results = {'error': str(e)}
        
        return results
    
    def run_lightgbm_classification(self, data: Dict) -> Dict:
        """Run LightGBM classification with hyperparameter tuning."""
        if not self.config['models']['lightgbm']['enabled']:
            return {}
        
        self.logger.info("=" * 60)
        self.logger.info("LIGHTGBM CLASSIFICATION")
        self.logger.info("=" * 60)
        
        self.experiment_tracker.start_model_training("LightGBM", self.config['models']['lightgbm'])
        
        results = {}
        
        try:
            # Get hyperparameters
            hyperparams = self.config['models']['lightgbm']['hyperparameters']
            
            # Simple grid search for best parameters
            best_score = 0
            best_params = {}
            best_model = None
            
            X_train = data['train']['X']
            y_train = data['train']['y']

            # for n_est, lr, max_depth, num_leaves in zip(hyperparams['n_estimators'], hyperparams['learning_rate'], hyperparams['max_depth'], hyperparams['num_leaves']):
            #     self.logger.info(f"Current params: n_estimators={n_est}, learning_rate={lr}, max_depth={max_depth}, num_leaves={num_leaves}")
            
            for n_est in hyperparams['n_estimators']:
                for lr in hyperparams['learning_rate']:
                    for max_depth in hyperparams['max_depth']:
                        for num_leaves in hyperparams['num_leaves']:

                            self.logger.info(f"Current params: n_estimators={n_est}, learning_rate={lr}, max_depth={max_depth}, num_leaves={num_leaves}")
                            # Create model
                            model = lgb.LGBMClassifier(
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=max_depth,
                                num_leaves=num_leaves,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                n_jobs=-1,
                                verbose=-1
                            )
                            
                            # Cross-validation or simple training
                            if self.config['experiment']['evaluation']['cross_validate']:
                                cv = StratifiedKFold(n_splits=self.config['dataset']['splitting']['cv_folds'], 
                                                    shuffle=True, random_state=42)
                                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                                mean_score = scores.mean()
                            else:
                                model.fit(X_train, y_train)
                                mean_score = model.score(X_train, y_train)
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = {
                                    'n_estimators': n_est, 'learning_rate': lr, 
                                    'max_depth': max_depth, 'num_leaves': num_leaves
                                }
                                if not self.config['experiment']['evaluation']['cross_validate']:
                                    best_model = model
            
            self.logger.info(f"Best LightGBM parameters: {best_params}")
            self.logger.info(f"Best CV score: {best_score:.4f}")
            
            # Train final model with best parameters
            if best_model is None:
                best_model = lgb.LGBMClassifier(**best_params, subsample=0.8, colsample_bytree=0.8,
                                               random_state=42, n_jobs=-1, verbose=-1)
                best_model.fit(X_train, y_train)
            
            # Evaluate on all splits
            for split_name in ['train', 'validation', 'test']:
                if split_name in data:
                    X = data[split_name]['X']
                    y = data[split_name]['y']
                    
                    y_pred = best_model.predict(X)
                    y_prob = best_model.predict_proba(X)[:, 1]
                    
                    metrics = self.experiment_tracker.evaluate_model(
                        "LightGBM", y, y_pred, y_prob, split_name
                    )
                    results[split_name] = metrics
                    
                    # Create visualizations for test set
                    if split_name == 'test':
                        self.experiment_tracker.create_confusion_matrix_plot("LightGBM", y, y_pred)
                        self.experiment_tracker.create_roc_curve_plot("LightGBM", y, y_prob)
                        self.experiment_tracker.create_precision_recall_curve_plot("LightGBM", y, y_prob)
            
            self.experiment_tracker.finish_model_training("LightGBM", best_model, results.get('test', {}))
            
        except Exception as e:
            self.logger.error(f"LightGBM classification failed: {e}")
            results = {'error': str(e)}
        
        return results
    
    def run_deep_learning_classification(self, data: Dict) -> Dict:
        """Run deep learning classification with multiple architectures."""
        if not self.config['models']['deep_learning']['enabled']:
            return {}
        
        self.logger.info("=" * 60)
        self.logger.info("DEEP LEARNING CLASSIFICATION")
        self.logger.info("=" * 60)
        
        # Import deep learning components
        from deep_learning_classifier import (
            create_model, train_model, evaluate_model as dl_evaluate_model
        )
        from torch.utils.data import DataLoader, TensorDataset
        
        results = {}
        dl_config = self.config['models']['deep_learning']
        
        # Prepare data loaders
        train_loader = self._create_data_loader(data['train'], dl_config['training']['batch_size'], shuffle=True)
        val_loader = self._create_data_loader(data['validation'], dl_config['training']['batch_size'], shuffle=False) if 'validation' in data else None
        test_loader = self._create_data_loader(data['test'], dl_config['training']['batch_size'], shuffle=False) if 'test' in data else None
        
        input_size = data['train']['X'].shape[1]
        device = torch.device(self.config['hardware']['device'] if self.config['hardware']['device'] != 'auto' 
                             else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Train each enabled architecture
        for arch_name, arch_config in dl_config['architectures'].items():
            if not arch_config['enabled']:
                continue
            
            model_name = f"DL_{arch_name.upper()}"
            self.logger.info(f"Training {model_name}...")
            
            # try:
            self.experiment_tracker.start_model_training(model_name, arch_config)
            
            # Create model
            model = create_model(arch_name, arch_config, input_size)
            
            # Train model
            trained_model, history = train_model(model, train_loader, val_loader or test_loader, 
                                                {**self.config, 'hardware': self.config['hardware']})
            
            # Log training history
            for epoch, metrics in enumerate(history['train_loss']):
                epoch_metrics = {
                    'train_loss': history['train_loss'][epoch],
                    'val_loss': history['val_loss'][epoch] if epoch < len(history['val_loss']) else None,
                    'train_acc': history['train_acc'][epoch] if epoch < len(history['train_acc']) else None,
                    'val_acc': history['val_acc'][epoch] if epoch < len(history['val_acc']) else None,
                }
                self.experiment_tracker.log_training_metrics(model_name, epoch, epoch_metrics)
            
            # Evaluate model
            model_results = {}
            for split_name, data_loader in [('train', train_loader), ('test', test_loader)]:
                if data_loader is not None:
                    accuracy, cm, report, y_pred, y_true = dl_evaluate_model(trained_model, data_loader, device)
                    
                    # Convert predictions to probabilities (assuming sigmoid output)
                    trained_model.eval()
                    y_prob = []
                    with torch.no_grad():
                        for batch_x, _ in data_loader:
                            batch_x = batch_x.to(device)
                            outputs = trained_model(batch_x)
                            y_prob.extend(outputs.squeeze().cpu().numpy())
                    
                    y_prob = np.array(y_prob)
                    
                    metrics = self.experiment_tracker.evaluate_model(
                        model_name, y_true, y_pred, y_prob, split_name
                    )
                    model_results[split_name] = metrics
                    
                    # Create visualizations for test set
                    if split_name == 'test':
                        self.experiment_tracker.create_confusion_matrix_plot(model_name, y_true, y_pred)
                        self.experiment_tracker.create_roc_curve_plot(model_name, y_true, y_prob)
                        self.experiment_tracker.create_precision_recall_curve_plot(model_name, y_true, y_prob)
            
            # Create training history plot
            self.experiment_tracker.create_training_history_plot(model_name)
            
            self.experiment_tracker.finish_model_training(model_name, trained_model, model_results.get('test', {}))
            
            results[model_name] = model_results
                
            # except Exception as e:
            #     self.logger.error(f"{model_name} training failed: {e}")
            #     results[model_name] = {'error': str(e)}
        
        return results
    
    def _create_data_loader(self, data_split: Dict, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Create PyTorch DataLoader from data split."""
        X = torch.FloatTensor(data_split['X'].values)
        y = torch.FloatTensor(data_split['y'].values)
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=self.config['hardware']['num_workers'])
    
    def run_experiment(self) -> Dict:
        """Run the complete multi-dataset classification experiment."""
        self.logger.info("🚀 Starting Multi-Dataset Fog Gaussian Classification Experiment")
        
        # Prepare datasets
        data = self.prepare_datasets()
        
        # Track all results
        all_results = {}
        
        # Run each method
        methods = self.config['experiment']['methods']
        if 'all' in methods:
            methods = ['svm', 'lightgbm', 'deep_learning']
        
        if 'svm' in methods:
            svm_results = self.run_svm_classification(data)
            all_results['svm'] = svm_results
        
        if 'lightgbm' in methods:
            lgbm_results = self.run_lightgbm_classification(data)
            all_results['lightgbm'] = lgbm_results
        
        if 'deep_learning' in methods:
            dl_results = self.run_deep_learning_classification(data)
            all_results['deep_learning'] = dl_results
        
        # Finalize experiment
        self.experiment_tracker.finalize_experiment()
        
        return all_results
    
    def print_final_summary(self, results: Dict):
        """Print final experiment summary."""
        self.experiment_tracker.print_summary()
        
        print(f"\n EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f" Results directory: {self.experiment_tracker.experiment_dir}")
        print(f" Best model: {self.experiment_tracker.get_best_model('accuracy')}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-Dataset Fog Gaussian Classification')
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                       help='Path to experiment configuration file')
    parser.add_argument('--data-dir', type=str, help='Override data directory from config')
    parser.add_argument('--methods', nargs='+', choices=['svm', 'lightgbm', 'deep_learning', 'all'],
                       help='Override methods to run')
    parser.add_argument('--experiment-name', type=str, help='Override experiment name')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Please create the configuration file or use --config to specify a different path.")
        sys.exit(1)
    
    try:
        # Create runner
        runner = MultiDatasetClassificationRunner(args.config)
        
        # Override config with command line arguments
        if args.data_dir:
            runner.config['dataset']['data_directory'] = args.data_dir
        if args.methods:
            runner.config['experiment']['methods'] = args.methods
        if args.experiment_name:
            runner.config['experiment']['name'] = args.experiment_name
        
        # Run experiment
        results = runner.run_experiment()
        
        # Print summary
        runner.print_final_summary(results)
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        logging.exception("Experiment failed with exception:")
        sys.exit(1)

if __name__ == "__main__":
    main()