# KAN-based Atmospheric Parameter Mapping

This package implements a true **Kolmogorov-Arnold Network (KAN)** for mapping atmospheric scattering parameters (beta, alpha) to Gaussian splatting parameters.

## What Makes This KAN Special

KAN differs fundamentally from traditional MLPs:
- **No fixed activation functions** (ReLU, sigmoid, etc.)
- **Learnable univariate functions** on each edge φᵢⱼ(x)
- Based on the **Kolmogorov-Arnold theorem**: any multivariate function can be represented as compositions of univariate functions
- **Interpretable**: you can visualize what each edge learned
- **Efficient**: fewer parameters for smooth function approximation

## The Core Idea

In traditional neural networks: `y = σ(Wx + b)` (fixed σ)

In KAN: `yⱼ = Σᵢ φᵢⱼ(xᵢ)` (learnable φᵢⱼ using B-splines)

The system learns the mapping from atmospheric conditions to 3D Gaussian parameters through compositions of these learnable univariate functions.

## Files

- `kan_layer.py` - Core KAN layer implementation with B-spline basis functions
- `kan_model.py` - Main KAN model class for atmospheric parameter mapping
- `data_preprocessing.py` - Data loading and preprocessing utilities
- `train_kan.py` - Training script with validation and early stopping
- `inference.py` - Inference engine for making predictions
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_kan.py --data_path ../data/dataset_truck_beta6_alpha250.csv --epochs 200 --batch_size 64
```

### 3. Make Predictions

Single prediction:
```bash
python inference.py --model_path outputs/kan_atmospheric_model.pth --scaler_path outputs/kan_atmospheric_model_scalers.pkl --mode single --beta 6.0 --alpha 250.0
```

Parameter sweep:
```bash
python inference.py --model_path outputs/kan_atmospheric_model.pth --scaler_path outputs/kan_atmospheric_model_scalers.pkl --mode sweep --beta_min 1.0 --beta_max 10.0 --alpha_min 100.0 --alpha_max 500.0 --visualize
```

## Model Architecture

The KAN model consists of:
- **Input normalization layer**
- **KAN layers** with learnable univariate functions φᵢⱼ(x) on each edge
- Each φᵢⱼ is represented as a **B-spline with learnable coefficients**
- **No traditional activation functions** - the φᵢⱼ functions ARE the activations
- **Composition of layers**: f(x) = Φₗ ∘ Φₗ₋₁ ∘ ... ∘ Φ₁(x)

### Example Architecture:
```
Input (2D): [beta, alpha]
    ↓ φ₁,₁(beta) + φ₂,₁(alpha) → hidden₁
    ↓ φ₁,₂(beta) + φ₂,₂(alpha) → hidden₂
    ...
    ↓ φᵢ,ⱼ(hiddenᵢ) → output
Output (14D): [pos_x, pos_y, pos_z, ...]
```

## Data Format

Expected CSV columns:
- **Inputs**: `beta`, `alpha` - atmospheric scattering parameters
- **Outputs**: Gaussian splatting parameters:
  - Position: `pos_x`, `pos_y`, `pos_z`
  - Color: `f_dc_0`, `f_dc_1`, `f_dc_2`
  - Scale: `scale_x`, `scale_y`, `scale_z`
  - Rotation: `rot_0`, `rot_1`, `rot_2`, `rot_3`
  - Opacity: `opacity`

## Training Options

Key training parameters:
- `--hidden_dims`: Hidden layer sizes (default: [16, 8] - smaller for KAN!)
- `--grid_size`: B-spline grid resolution (default: 5)
- `--spline_order`: B-spline polynomial order (default: 3)
- `--noise_scale`: Parameter initialization scale (default: 0.1)
- `--lambda_sparse`: Sparsity regularization for function selection (default: 0.001)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: L2 regularization (default: 1e-5)
- `--loss_type`: Loss function (mse, mae, huber)
- `--patience`: Early stopping patience (default: 30)

## KAN-Specific Features

### Function Visualization
After training, you can visualize the learned univariate functions:
```python
model.visualize_learned_functions("./kan_functions/")
# Creates plots showing φᵢⱼ(x) for each edge
```

### Function Analysis
```python
# Analyze which functions are most important
importance = model.analyze_function_importance()

# Get function representations for further analysis
functions = model.get_function_representations(layer_idx=0)
```

## Example Usage in Python

```python
from data_preprocessing import AtmosphericDataPreprocessor
from kan_model import AtmosphericKANModel
from inference import KANInferenceEngine

# Training
preprocessor = AtmosphericDataPreprocessor()
datasets = preprocessor.create_datasets('data.csv')
model = AtmosphericKANModel(input_dim=2, output_dim=14)
results = model.train(datasets['train'], datasets['val'])

# Inference
engine = KANInferenceEngine('model.pth', 'scalers.pkl')
prediction = engine.predict_single(beta=6.0, alpha=250.0)
```

## Why KAN for Atmospheric Modeling?

The KAN model is particularly suited for this task because:

1. **Smooth Function Approximation**: Atmospheric scattering typically involves smooth, continuous relationships
2. **Interpretability**: You can visualize how beta and alpha individually affect each Gaussian parameter
3. **Efficient Representation**: Fewer parameters needed compared to MLPs for smooth functions
4. **Function Decomposition**: Complex atmospheric effects decomposed into understandable univariate components

## Model Performance

The KAN model provides:
- **Learnable nonlinear functions** through B-spline representations
- **Superior interpretability** - you can see exactly what each function learned
- **Efficient parameter usage** for smooth multivariate function approximation
- **Robust performance** on atmospheric parameter interpolation
- **No activation function assumptions** - the model learns the optimal nonlinearities

## Output Files

Training produces:
- `model.pth` - Trained KAN model weights
- `scalers.pkl` - Data normalization parameters
- `training_history.png` - Loss curves
- `learned_functions/` - **Visualizations of all learned φᵢⱼ functions**
- `function_importance.json` - **Analysis of which functions are most active**
- `results.json` - Training metrics and configuration

The learned function visualizations are unique to KAN and show you exactly how the model represents the atmospheric mapping!