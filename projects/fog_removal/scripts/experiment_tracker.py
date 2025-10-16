#!/usr/bin/env python3
"""
Experiment Tracker for Multi-Dataset Fog Gaussian Classification
Handles logging, metrics tracking, and result aggregation
"""

import os
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

class ExperimentTracker:
    """Tracks experiments, metrics, and results across multiple models and datasets."""
    
    def __init__(self, experiment_name: str, output_directory: str = "experiments"):
        self.experiment_name = experiment_name
        self.output_directory = parent_directory / Path(output_directory)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_directory / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.plots_dir = self.experiment_dir / "plots"
        self.results_dir = self.experiment_dir / "results"
        self.logs_dir = self.experiment_dir / "logs"
        
        for directory in [self.models_dir, self.plots_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.experiment_info = {
            'name': experiment_name,
            'timestamp': timestamp,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'models': {},
            'datasets': {},
            'results': {}
        }
        
        self.results = []
        self.models = {}
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Experiment '{experiment_name}' initialized")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / "experiment.log"
        
        # Create logger
        self.logger = logging.getLogger(f"experiment_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False 
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_dataset_info(self, dataset_info: Dict):
        """Log dataset information."""
        self.experiment_info['datasets'] = dataset_info
        self.logger.info(f"Dataset info logged: {len(dataset_info.get('source_datasets', []))} datasets")
    
    def start_model_training(self, model_name: str, model_config: Dict = None):
        """Start tracking a model training."""
        self.experiment_info['models'][model_name] = {
            'config': model_config or {},
            'start_time': datetime.now().isoformat(),
            'status': 'training',
            'metrics': {},
            'training_history': []
        }
        
        self.logger.info(f"Started training {model_name}")
    
    def log_training_metrics(self, model_name: str, epoch: int, metrics: Dict):
        """Log training metrics for a specific epoch."""
        if model_name not in self.experiment_info['models']:
            self.start_model_training(model_name)
        
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.experiment_info['models'][model_name]['training_history'].append(metrics_with_epoch)
    
    def finish_model_training(self, model_name: str, model_object: Any = None, 
                            final_metrics: Dict = None):
        """Finish tracking a model training."""
        if model_name not in self.experiment_info['models']:
            self.start_model_training(model_name)
        
        self.experiment_info['models'][model_name].update({
            'end_time': datetime.now().isoformat(),
            'status': 'completed',
            'final_metrics': final_metrics or {}
        })
        
        if model_object is not None:
            self.models[model_name] = model_object
            
            # Save model
            model_path = self.models_dir / f"{model_name}_model.pkl"
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model_object, f)
                self.logger.info(f"Model {model_name} saved to {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to save model {model_name}: {e}")
        
        self.logger.info(f"Finished training {model_name}")
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_prob: Optional[np.ndarray] = None, dataset_split: str = "test") -> Dict:
        """Evaluate a model and return metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # AUC if probabilities are available
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                    metrics['auc'] = roc_auc_score(y_true, y_prob)
                else:
                    metrics['auc'] = np.nan
            except Exception as e:
                self.logger.warning(f"Could not compute AUC for {model_name}: {e}")
                metrics['auc'] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log results
        result_entry = {
            'model_name': model_name,
            'dataset_split': dataset_split,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'n_samples': len(y_true),
            'n_positive': int(np.sum(y_true)),
            'n_negative': int(len(y_true) - np.sum(y_true))
        }
        
        self.results.append(result_entry)
        
        # Update experiment info
        if model_name in self.experiment_info['models']:
            self.experiment_info['models'][model_name]['final_metrics'] = metrics
        
        self.logger.info(f"Model {model_name} evaluated - Accuracy: {metrics['accuracy']:.4f}, "
                        f"F1: {metrics['f1']:.4f}, AUC: {metrics.get('auc', 'N/A')}")
        
        return metrics
    
    def create_confusion_matrix_plot(self, model_name: str, y_true: np.ndarray, 
                                   y_pred: np.ndarray, title: str = None):
        """Create and save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Fog', 'Fog'],
                   yticklabels=['No Fog', 'Fog'])
        
        accuracy = accuracy_score(y_true, y_pred)
        plot_title = title or f'{model_name} Confusion Matrix (Acc: {accuracy:.4f})'
        plt.title(plot_title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plot_path = self.plots_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix plot saved: {plot_path}")
    
    def create_roc_curve_plot(self, model_name: str, y_true: np.ndarray, 
                            y_prob: np.ndarray, title: str = None):
        """Create and save ROC curve plot."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            
            plot_title = title or f'{model_name} ROC Curve'
            plt.title(plot_title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = self.plots_dir / f"{model_name}_roc_curve.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"ROC curve plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create ROC curve for {model_name}: {e}")
    
    def create_precision_recall_curve_plot(self, model_name: str, y_true: np.ndarray, 
                                         y_prob: np.ndarray, title: str = None):
        """Create and save precision-recall curve plot."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2, label=f'{model_name}')
            
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            
            plot_title = title or f'{model_name} Precision-Recall Curve'
            plt.title(plot_title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = self.plots_dir / f"{model_name}_pr_curve.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"PR curve plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create PR curve for {model_name}: {e}")
    
    def create_training_history_plot(self, model_name: str, title: str = None):
        """Create training history plot for deep learning models."""
        if model_name not in self.experiment_info['models']:
            return
        
        history = self.experiment_info['models'][model_name].get('training_history', [])
        if not history:
            return
        
        try:
            df_history = pd.DataFrame(history)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss plot
            if 'train_loss' in df_history.columns and 'val_loss' in df_history.columns:
                axes[0].plot(df_history['epoch'], df_history['train_loss'], label='Train Loss')
                axes[0].plot(df_history['epoch'], df_history['val_loss'], label='Validation Loss')
                axes[0].set_title(f'{model_name} - Training Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            if 'train_acc' in df_history.columns and 'val_acc' in df_history.columns:
                axes[1].plot(df_history['epoch'], df_history['train_acc'], label='Train Accuracy')
                axes[1].plot(df_history['epoch'], df_history['val_acc'], label='Validation Accuracy')
                axes[1].set_title(f'{model_name} - Training Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = self.plots_dir / f"{model_name}_training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training history plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create training history plot for {model_name}: {e}")
    
    def create_model_comparison_plot(self, metric: str = 'accuracy'):
        """Create model comparison plot."""
        if not self.results:
            return
        
        # Get test results
        test_results = [r for r in self.results if r['dataset_split'] == 'test']
        
        if not test_results:
            return
        
        model_names = [r['model_name'] for r in test_results]
        metric_values = [r['metrics'].get(metric, 0) for r in test_results]
        
        # Sort by metric value
        sorted_data = sorted(zip(model_names, metric_values), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_values = zip(*sorted_data)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_names, sorted_values, color='skyblue', alpha=0.8)
        
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1 if metric != 'auc' or max(sorted_values) <= 1 else max(sorted_values) * 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, sorted_values):
            if not np.isnan(value):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = self.plots_dir / f"model_comparison_{metric}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model comparison plot saved: {plot_path}")
    
    def save_results(self):
        """Save all results and experiment info."""
        # Save experiment info
        experiment_file = self.results_dir / "experiment_info.json"
        with open(experiment_file, 'w') as f:
            json.dump(self.experiment_info, f, indent=2, default=str)
        
        # Save detailed results
        results_file = self.results_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save results as CSV
        if self.results:
            df_results = self._create_results_dataframe()
            csv_file = self.results_dir / "results_summary.csv"
            df_results.to_csv(csv_file, index=False)
            
            self.logger.info(f"Results saved to {csv_file}")
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame of results."""
        summary_data = []
        
        for result in self.results:
            if result['dataset_split'] == 'test':  # Only test results for summary
                row = {
                    'model_name': result['model_name'],
                    'accuracy': result['metrics'].get('accuracy', np.nan),
                    'precision': result['metrics'].get('precision', np.nan),
                    'recall': result['metrics'].get('recall', np.nan),
                    'f1_score': result['metrics'].get('f1', np.nan),
                    'auc': result['metrics'].get('auc', np.nan),
                    'n_samples': result['n_samples'],
                    'n_positive': result['n_positive'],
                    'n_negative': result['n_negative']
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def create_experiment_report(self):
        """Create a comprehensive HTML experiment report."""
        try:
            html_content = self._generate_html_report()
            
            report_file = self.results_dir / "experiment_report.html"
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Experiment report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment report: {e}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        # Get summary statistics
        if self.results:
            df_results = self._create_results_dataframe()
            best_model = df_results.loc[df_results['accuracy'].idxmax()] if len(df_results) > 0 else None
        else:
            df_results = pd.DataFrame()
            best_model = None
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report: {self.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #2c3e50; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Experiment Report: {self.experiment_name}</h1>
                <p><strong>Start Time:</strong> {self.experiment_info['start_time']}</p>
                <p><strong>Experiment Directory:</strong> {self.experiment_dir}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Information</h2>
                <ul>
        """
        
        dataset_info = self.experiment_info.get('datasets', {})
        if dataset_info:
            html += f"""
                    <li><strong>Total Samples:</strong> {dataset_info.get('total_samples', 'N/A')}</li>
                    <li><strong>Train Samples:</strong> {dataset_info.get('train_samples', 'N/A')}</li>
                    <li><strong>Validation Samples:</strong> {dataset_info.get('val_samples', 'N/A')}</li>
                    <li><strong>Test Samples:</strong> {dataset_info.get('test_samples', 'N/A')}</li>
                    <li><strong>Source Datasets:</strong> {len(dataset_info.get('source_datasets', []))}</li>
            """
        
        html += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Model Results</h2>
        """
        
        if not df_results.empty:
            html += "<table><tr>"
            for col in df_results.columns:
                html += f"<th>{col.replace('_', ' ').title()}</th>"
            html += "</tr>"
            
            for _, row in df_results.iterrows():
                html += "<tr>"
                for col in df_results.columns:
                    value = row[col]
                    if isinstance(value, float) and not np.isnan(value):
                        if col in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
                            html += f"<td class='metric'>{value:.4f}</td>"
                        else:
                            html += f"<td>{value}</td>"
                    else:
                        html += f"<td>{value}</td>"
                html += "</tr>"
            
            html += "</table>"
        else:
            html += "<p>No results available.</p>"
        
        if best_model is not None:
            html += f"""
            <div class="section">
                <h2>Best Model</h2>
                <p><strong>Model:</strong> {best_model['model_name']}</p>
                <p><strong>Accuracy:</strong> <span class="metric">{best_model['accuracy']:.4f}</span></p>
                <p><strong>F1 Score:</strong> <span class="metric">{best_model['f1_score']:.4f}</span></p>
                <p><strong>AUC:</strong> <span class="metric">{best_model['auc']:.4f if not np.isnan(best_model['auc']) else 'N/A'}</span></p>
            </div>
            """
        
        html += """
            <div class="section">
                <h2>Files Generated</h2>
                <ul>
                    <li><strong>Models:</strong> models/ directory</li>
                    <li><strong>Plots:</strong> plots/ directory</li>
                    <li><strong>Results:</strong> results/ directory</li>
                    <li><strong>Logs:</strong> logs/ directory</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def finalize_experiment(self):
        """Finalize the experiment and save all results."""
        self.experiment_info['end_time'] = datetime.now().isoformat()
        self.experiment_info['status'] = 'completed'
        
        # Save all results
        self.save_results()
        
        # Create comparison plots
        for metric in ['accuracy', 'f1', 'auc']:
            self.create_model_comparison_plot(metric)
        
        # Create experiment report
        self.create_experiment_report()
        
        self.logger.info(f"Experiment '{self.experiment_name}' finalized")
        self.logger.info(f"Results available in: {self.experiment_dir}")
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[str]:
        """Get the name of the best performing model."""
        if not self.results:
            return None
        
        test_results = [r for r in self.results if r['dataset_split'] == 'test']
        
        if not test_results:
            return None
        
        best_result = max(test_results, key=lambda x: x['metrics'].get(metric, 0))
        return best_result['model_name']
    
    def print_summary(self):
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*60}")
        
        print(f"Experiment Directory: {self.experiment_dir}")
        
        if self.results:
            df_results = self._create_results_dataframe()
            print(f"\nModels Evaluated: {len(df_results)}")
            
            if not df_results.empty:
                print(f"\nResults Summary:")
                print(df_results.round(4).to_string(index=False))
                
                best_model = self.get_best_model('accuracy')
                if best_model:
                    best_acc = df_results[df_results['model_name'] == best_model]['accuracy'].iloc[0]
                    print(f"\nBest Model: {best_model} (Accuracy: {best_acc:.4f})")
        
        print(f"{'='*60}")

def main():
    """Example usage of ExperimentTracker."""
    # Create tracker
    tracker = ExperimentTracker("test_experiment")
    
    # Simulate model training and evaluation
    tracker.start_model_training("test_model", {"param1": 1, "param2": 2})
    
    # Simulate evaluation
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_prob = np.random.random(1000)
    
    tracker.evaluate_model("test_model", y_true, y_pred, y_prob)
    tracker.finish_model_training("test_model")
    
    # Create visualizations
    tracker.create_confusion_matrix_plot("test_model", y_true, y_pred)
    tracker.create_roc_curve_plot("test_model", y_true, y_prob)
    
    # Finalize
    tracker.finalize_experiment()
    tracker.print_summary()

if __name__ == "__main__":
    main()