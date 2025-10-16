# Enhanced Multi-Dataset Fog Gaussian Classification

This repository contains a comprehensive system for classifying fog Gaussians using multiple machine learning approaches across multiple datasets. The project is based on the `fog_removal.ipynb` notebook and implements an advanced pipeline with:

1. **Automatic dataset discovery and validation**
2. **Multiple data splitting strategies**  
3. **Three types of classification methods** (SVM, LightGBM, Deep Learning)
4. **Comprehensive experiment tracking and reporting**

## 🆕 New Features

- **Multi-dataset support**: Automatically discover and process multiple CSV files
- **Smart data splitting**: Combined, cross-dataset, and parameter-based splitting strategies
- **Experiment tracking**: Comprehensive logging, metrics tracking, and HTML reports
- **Advanced configuration**: YAML-based configuration system
- **Cross-validation support**: Built-in cross-validation and hyperparameter tuning
- **Rich visualizations**: Automatic generation of plots and comparison charts

## 📁 Enhanced Project Structure

```
scripts/
├── config/
│   ├── experiment_config.yaml    # Complete experiment configuration
│   └── dl_config.yaml            # Legacy deep learning config (still supported)
├── data/
│   └── *.csv                     # Multiple CSV datasets (auto-discovered)
├── cache/                        # Dataset validation cache (auto-created)
├── experiments/                  # Experiment results (auto-created)
│   └── experiment_name_timestamp/
│       ├── models/               # Trained models
│       ├── plots/                # All visualizations
│       ├── results/              # Results and reports
│       └── logs/                 # Experiment logs
├── dataset_manager.py            # Multi-dataset management
├── experiment_tracker.py         # Experiment tracking system
├── enhanced_run_classification.py # Main enhanced runner
├── svm_classifier.py             # SVM implementation
├── lightgbm_classifier.py        # LightGBM implementation
├── deep_learning_classifier.py   # Deep learning models
└── README.md                     # This file
```

## 🔧 Requirements

Install the required packages:

```bash
pip install numpy pandas scikit-learn lightgbm torch matplotlib seaborn pyyaml tqdm joblib
```

## 📊 Dataset Format

The CSV dataset should contain 3D Gaussian features with the following structure:
- **Position**: `pos_x`, `pos_y`, `pos_z`
- **Spherical Harmonics**: `f_dc_0`, `f_dc_1`, `f_dc_2`
- **Scale**: `scale_x`, `scale_y`, `scale_z`
- **Rotation**: `rot_0`, `rot_1`, `rot_2`, `rot_3`
- **Opacity**: `opacity`
- **Parameters**: `beta`, `alpha`
- **Target**: `is_fog` (0 or 1)
- **Additional features**: Various spatial and neighborhood features

## 🚀 Quick Start

### 1. Enhanced Multi-Dataset Runner (Recommended)
```bash
# Run with default configuration (processes all CSV files in data/ directory)
python enhanced_run_classification.py

# Specify custom data directory
python enhanced_run_classification.py --data-dir /path/to/your/csv/files

# Run specific methods only
python enhanced_run_classification.py --methods svm lightgbm

# Use custom configuration
python enhanced_run_classification.py --config config/my_experiment.yaml

# Override experiment name
python enhanced_run_classification.py --experiment-name my_fog_experiment
```

### 2. Dataset Manager (Standalone)
```bash
# Discover and validate datasets
python dataset_manager.py --data-dir data/ --strategy combined

# Different splitting strategies
python dataset_manager.py --data-dir data/ --strategy cross_dataset
python dataset_manager.py --data-dir data/ --strategy parameter_based
```

### 3. Legacy Single-Dataset Scripts (Still Supported)
```bash
# SVM
python svm_classifier.py

# LightGBM
python lightgbm_classifier.py

# Deep Learning (all 6 models)
python deep_learning_classifier.py
```

## 📈 Methods Overview

### 1. Support Vector Machine (SVM)
- **File**: `svm_classifier.py`
- **Features**: RBF kernel, standardized features
- **Outputs**: Model, scaler, predictions, confusion matrix

### 2. LightGBM
- **File**: `lightgbm_classifier.py`
- **Features**: Gradient boosting, feature importance analysis
- **Outputs**: Model, predictions, feature importance plots

### 3. Deep Learning Models
- **File**: `deep_learning_classifier.py`
- **Models Implemented**:
  - **MLP**: Multi-layer perceptron with configurable layers
  - **Transformer**: Transformer encoder with self-attention
  - **CNN1D**: 1D convolutional network
  - **LSTM**: LSTM with bidirectional option
  - **Self-Attention**: Multi-head self-attention network
  - **ResNet**: Residual network with skip connections

## ⚙️ Enhanced Configuration System

The system uses a comprehensive YAML configuration file (`config/experiment_config.yaml`):

### Dataset Configuration
```yaml
dataset:
  data_directory: "data"
  splitting:
    strategy: "combined"  # Options: "combined", "cross_dataset", "parameter_based"
    test_size: 0.2
    validation_size: 0.1
  
  # Feature Selection Configuration
  features:
    selection_mode: "manual"  # Options: "auto", "manual", "exclude", "statistical", "categories"
    manual_features: [
      "f_dc_0", "f_dc_1", "f_dc_2",      # Spherical harmonics
      "scale_x", "scale_y", "scale_z",    # Scale parameters  
      "opacity",                          # Opacity
      "knn_6_mean_distance", "knn_6_density",  # Neighborhood features
      "local_linearity", "local_planarity"     # Geometric features
    ]
  
  preprocessing:
    normalize_features: true
    handle_missing: "drop"
```

### Experiment Configuration  
```yaml
experiment:
  name: "fog_gaussian_classification"
  methods: ["svm", "lightgbm", "deep_learning"]
  evaluation:
    cross_validate: true
    metrics: ["accuracy", "precision", "recall", "f1", "auc"]
```

### Model Configurations
```yaml
models:
  svm:
    enabled: true
    hyperparameters:
      kernel: ["rbf", "linear"]
      C: [0.1, 1.0, 10.0]
      
  lightgbm:
    enabled: true  
    hyperparameters:
      n_estimators: [100, 200, 300]
      learning_rate: [0.05, 0.1, 0.15]
      
  deep_learning:
    enabled: true
    architectures:
      mlp:
        enabled: true
        hidden_layers: [1024, 512, 256, 128, 32]
```

### Advanced Features
```yaml
advanced:
  balancing:
    enabled: false
    method: "smote"
  feature_engineering:
    enabled: false
    polynomial_features: false
  ensemble:
    enabled: false
    methods: ["voting", "stacking"]
```

## 📊 Enhanced Output Structure

Each experiment creates a timestamped directory with comprehensive results:

```
experiments/experiment_name_20240316_143052/
├── models/                    # Trained models
│   ├── SVM_model.pkl
│   ├── LightGBM_model.pkl
│   └── DL_*.pth              # Deep learning models
├── plots/                     # All visualizations
│   ├── *_confusion_matrix.png
│   ├── *_roc_curve.png
│   ├── *_pr_curve.png
│   ├── *_training_history.png
│   ├── *_feature_importance.png
│   └── model_comparison_*.png
├── results/                   # Results and reports
│   ├── experiment_info.json
│   ├── detailed_results.json
│   ├── results_summary.csv
│   └── experiment_report.html # Comprehensive HTML report
└── logs/                      # Experiment logs
    └── experiment.log
```

### Key Output Files
- **HTML Report**: Complete experiment summary with all metrics and visualizations
- **CSV Summary**: Easy-to-analyze results table
- **JSON Results**: Detailed machine-readable results
- **Model Files**: All trained models saved for future use
- **Visualizations**: Publication-ready plots and charts

## 🔍 Flexible Feature Selection System

The enhanced system provides multiple ways to select features for training:

### 1. Feature Selection Modes

#### **Auto Mode** (Default)
- Automatically excludes position coordinates and rotation parameters
- Uses all other available features
- Good starting point for most experiments

#### **Manual Mode** 
- Specify exact features to use in YAML configuration
- Complete control over feature selection
- Ideal for targeted experiments

#### **Exclude Mode**
- Specify features to exclude, use everything else
- Good for removing specific problematic features
- Maintains most features while filtering out unwanted ones

#### **Statistical Mode**
- Automatic feature selection based on statistical methods
- Options: mutual information, chi-squared, F-statistics, variance, random forest
- Selects top-k features based on relevance scores

#### **Categories Mode**
- Select features by predefined categories
- Mix and match different feature types
- Easy to experiment with different feature combinations

### 2. Available Feature Categories

- **Basic**: Core Gaussian properties (`f_dc_0`, `f_dc_1`, `f_dc_2`, `scale_x`, `scale_y`, `scale_z`, `opacity`)
- **Spatial**: Position and height features (`pos_x`, `pos_y`, `pos_z`, `distance_from_center`, `height_percentile`)  
- **Rotation**: Rotation parameters (`rot_0`, `rot_1`, `rot_2`, `rot_3`)
- **Neighborhood**: k-NN and radius-based features (density, distances, neighbor properties)
- **Geometric**: Local shape features (`local_linearity`, `local_planarity`, `local_sphericity`)
- **Context**: Scale ratios with neighbors

### 3. Example Feature Selection Configurations

#### Manual Selection (Core Features Only)
```yaml
features:
  selection_mode: "manual"
  manual_features: ["f_dc_0", "f_dc_1", "f_dc_2", "scale_x", "scale_y", "scale_z", "opacity"]
```

#### Category-Based Selection
```yaml
features:
  selection_mode: "categories"
  use_categories: ["basic", "neighborhood", "geometric"]
```

#### Statistical Selection
```yaml
features:
  selection_mode: "statistical" 
  statistical_selection:
    method: "mutual_info"
    k_best: 15
```

#### Exclusion-Based Selection
```yaml
features:
  selection_mode: "exclude"
  exclude_features: ["pos_x", "pos_y", "pos_z", "rot_0", "rot_1", "rot_2", "rot_3"]
```

## 📝 Advanced Usage Examples

### Multi-Dataset Experiment
```python
from dataset_manager import DatasetManager
from experiment_tracker import ExperimentTracker
from enhanced_run_classification import MultiDatasetClassificationRunner

# Create and run experiment
runner = MultiDatasetClassificationRunner('config/experiment_config.yaml')
results = runner.run_experiment()
```

### Dataset Management
```python
# Discover and validate datasets
dm = DatasetManager('data/')
dm.load_and_validate_all()
dm.print_summary()

# Create data splits with different strategies
splits = dm.create_dataset_splits(strategy='parameter_based')
```

### Experiment Tracking
```python
# Create experiment tracker
tracker = ExperimentTracker('my_experiment')

# Track model training
tracker.start_model_training('MyModel')
tracker.evaluate_model('MyModel', y_true, y_pred, y_prob)
tracker.finish_model_training('MyModel', model_object)

# Generate comprehensive report
tracker.finalize_experiment()
```

### Load Results from Previous Experiments
```python
import json
import pandas as pd

# Load experiment results
with open('experiments/my_exp_20240316_143052/results/detailed_results.json') as f:
    results = json.load(f)

# Load results summary
df = pd.read_csv('experiments/my_exp_20240316_143052/results/results_summary.csv')
print(df.sort_values('accuracy', ascending=False))
```

### Feature Selection Examples
```python
from feature_selector import FeatureSelector
import yaml

# Load configuration and test feature selection
with open('config/experiment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

selector = FeatureSelector(config)
selected_features = selector.select_features(df, 'is_fog')

# Get feature importance scores
importance_df = selector.get_feature_importance_summary()
print(importance_df.head(10))
```

### Custom Configuration

1. Copy `config/experiment_config.yaml` to create your own config
2. Modify dataset paths, model parameters, **feature selection**, splitting strategies, etc.
3. Run with: `python enhanced_run_classification.py --config my_config.yaml`

### Pre-built Configuration Examples

The system includes several ready-to-use configuration examples:

```bash
# Basic features only (core Gaussian properties)
python enhanced_run_classification.py --config config/experiment_config_basic_features.yaml

# Neighborhood and spatial features
python enhanced_run_classification.py --config config/experiment_config_neighborhood_features.yaml

# Statistical feature selection
python enhanced_run_classification.py --config config/experiment_config_statistical_selection.yaml
```

## 🎯 Performance Metrics

All methods provide:
- **Accuracy scores** (train/test)
- **Classification reports** (precision, recall, F1-score)
- **Confusion matrices**
- **ROC curves** (where applicable)

## 🔍 Data Splitting Strategies

The enhanced system supports three intelligent data splitting strategies:

### 1. Combined Strategy (Default)
- **Description**: Combines all datasets and splits randomly
- **Use case**: When you want maximum data utilization
- **Pros**: Large training set, good for model performance
- **Cons**: May not generalize well to new dataset conditions

### 2. Cross-Dataset Strategy
- **Description**: Uses different datasets for train/validation/test
- **Use case**: When you want to test generalization across datasets
- **Pros**: Tests true generalization capability
- **Cons**: Requires multiple datasets

### 3. Parameter-Based Strategy  
- **Description**: Splits based on beta/alpha parameter values
- **Use case**: When you want to ensure parameter diversity across splits
- **Pros**: Ensures model sees different fog conditions
- **Cons**: May create imbalanced splits

## 🐛 Enhanced Troubleshooting

### Common Issues

1. **Missing packages**: Run `pip install pyyaml` for additional dependencies
2. **CUDA issues**: Set `hardware.device: "cpu"` in experiment config
3. **Memory issues**: Reduce `batch_size` or enable `memory.chunk_size`
4. **No datasets found**: Check `dataset.data_directory` path in config
5. **Invalid datasets**: Check validation results in `cache/dataset_validation.json`

### Error Messages

- **"No valid datasets found"**: Ensure CSV files have required columns (`is_fog`, `pos_x`, etc.)
- **"Config file not found"**: Create `config/experiment_config.yaml` or specify with `--config`
- **"Experiment failed"**: Check logs in `experiments/*/logs/experiment.log`
- **"CUDA out of memory"**: Use smaller batch size or fewer models simultaneously

### Debug Mode
```bash
# Run with verbose logging
python enhanced_run_classification.py --config config/debug_config.yaml

# Check dataset validation
python dataset_manager.py --data-dir data/ 
```

## 🏆 Enhanced Model Selection

The enhanced system provides comprehensive model comparison:

### Automatic Evaluation
1. **Multi-metric evaluation**: Accuracy, Precision, Recall, F1, AUC
2. **Cross-validation**: Optional k-fold cross-validation for robust estimates  
3. **Hyperparameter tuning**: Grid search for optimal parameters
4. **Statistical significance**: Compare models across multiple datasets

### Selection Criteria
- **Accuracy**: Overall correctness (balanced datasets)
- **F1-score**: Harmonic mean of precision/recall (imbalanced datasets)
- **AUC**: Area under ROC curve (probability-based ranking)
- **Generalization**: Performance across different datasets/parameters
- **Computational cost**: Training time and inference speed

### Model Comparison Tools
- **HTML Reports**: Interactive comparison with all metrics
- **Comparison plots**: Bar charts showing relative performance
- **Statistical tests**: Significance testing across datasets
- **Ensemble options**: Combine multiple models for better performance

## 📚 References

Based on the research in fog removal using 3D Gaussians. The implementation follows the methodology from `fog_removal.ipynb` and extends it with additional deep learning architectures.

## 🤝 Contributing

### Adding New Models
1. **Deep Learning**: Add model class to `deep_learning_classifier.py` and update config
2. **Traditional ML**: Create new script following the pattern of existing classifiers
3. **Update configs**: Add model configuration to `experiment_config.yaml`
4. **Integration**: Update `enhanced_run_classification.py` to include new model

### Adding New Features
1. **Dataset processing**: Extend `dataset_manager.py`
2. **Experiment tracking**: Enhance `experiment_tracker.py`
3. **Visualization**: Add new plot types to experiment tracker
4. **Configuration**: Extend YAML configuration schema

### Testing
```bash
# Test with small dataset
python enhanced_run_classification.py --config config/test_config.yaml

# Test individual components
python dataset_manager.py --data-dir test_data/
python experiment_tracker.py  # Runs example
```

## 🎯 What's New vs Original

### Major Enhancements
- **🔄 Multi-dataset support**: Process dozens of CSV files automatically
- **🎛️ Smart data splitting**: Three strategies for different research needs  
- **📊 Rich experiment tracking**: HTML reports, comprehensive logging
- **⚙️ Advanced configuration**: YAML-based system for all parameters
- **📈 Enhanced visualizations**: Publication-ready plots and comparisons
- **🔍 Cross-validation**: Robust model evaluation across datasets
- **⚡ Hyperparameter tuning**: Automatic optimization for all models
- **🎨 Professional reporting**: HTML reports with all results

### Backward Compatibility
- **✅ Original scripts work**: All single-dataset scripts still function
- **✅ Same data format**: No changes needed to existing CSV files  
- **✅ Same models**: All original models (SVM, LightGBM, Deep Learning) included
- **✅ Easy migration**: Simple config file to enable new features

---

## 📋 Quick Migration Guide

**From single dataset to multi-dataset:**

1. **Organize your data**: Place all CSV files in a `data/` directory
2. **Create config**: Copy `config/experiment_config.yaml` and adjust paths
3. **Run enhanced version**: `python enhanced_run_classification.py`
4. **Review results**: Check HTML report in `experiments/` directory

**Note**: Ensure all CSV files have the required format with `is_fog` column (0 for no fog, 1 for fog) and standard 3D Gaussian features.