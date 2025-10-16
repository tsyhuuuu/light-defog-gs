# Fog Optimal Transport Analysis

This package provides comprehensive optimal transport (OT) analysis tools for Gaussian fog/haze data. It implements various OT-based methods to analyze spatial distribution, temporal evolution, and quality assessment of fog patterns.

## Features

### Core Analysis (`gaussian_ot_analysis.py`)
- **Spatial Clustering**: K-means clustering with OT distances between fog regions
- **Opacity Analysis**: Transport between high/low density fog areas
- **Temporal Evolution**: Track fog changes over time using OT distances
- **Transport Flow**: Analyze fog movement patterns between spatial quadrants  
- **Multi-feature Analysis**: Joint analysis of position, opacity, and scale parameters

### Applications (`applications.py`)
- **Density Estimation**: 3D fog density mapping using optimal transport
- **Anomaly Detection**: Identify unusual fog patterns based on OT distances
- **Pattern Classification**: OT-based clustering of fog configurations
- **Transport Optimization**: Plan optimal fog redistribution between regions
- **Quality Assessment**: Comprehensive fog quality scoring system

### Visualization (`visualization.py`)
- Interactive 3D plots with Plotly
- Static analysis plots with Matplotlib
- Transport flow diagrams
- Cluster analysis visualizations
- Multi-feature analysis plots

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from gaussian_ot_analysis import GaussianFogOTAnalysis
from visualization import FogOTVisualization
from applications import FogOTApplications

# Initialize analyzer
analyzer = GaussianFogOTAnalysis("path/to/train_fog_all.csv")

# Perform spatial analysis
clusters, distances, _, centers = analyzer.spatial_fog_clustering_ot()

# Analyze opacity patterns
opacity_result = analyzer.opacity_based_ot_analysis()

# Create visualizations
visualizer = FogOTVisualization(analyzer)
visualizer.save_all_plots("output_directory")

# Run applications
apps = FogOTApplications(analyzer)
quality = apps.fog_quality_assessment()
anomalies = apps.fog_anomaly_detection()
```

## Data Format

The input CSV should contain Gaussian parameters:
- `pos_x, pos_y, pos_z`: 3D position
- `scale_x, scale_y, scale_z`: Scale parameters
- `rot_0, rot_1, rot_2, rot_3`: Rotation quaternion
- `opacity`: Opacity/density value
- `f_dc_0, f_dc_1, f_dc_2`: Color coefficients
- `beta, alpha`: Additional fog parameters
- `is_fog`: Binary flag for fog vs non-fog
- `id`: Unique identifier

## Key Methods

### Wasserstein Distance Computation
```python
# Compare two fog regions
W_dist, transport_plan = analyzer.compute_gaussian_wasserstein_distance(
    subset1_indices, subset2_indices, feature='position'
)
```

### Fog Quality Assessment
```python
quality_metrics = applications.fog_quality_assessment()
# Returns: spatial_uniformity, density_consistency, temporal_stability, feature_coherence
```

### Anomaly Detection
```python
anomaly_result = applications.fog_anomaly_detection(contamination=0.1)
anomaly_indices = anomaly_result['anomaly_indices']
```

## Applications

### 1. Fog Simulation Quality Control
Assess the realism and quality of simulated fog using OT-based metrics.

### 2. Weather Pattern Analysis
Study real-world fog formation and dissipation patterns.

### 3. Computer Graphics
Optimize fog rendering and distribution in 3D scenes.

### 4. Environmental Monitoring
Analyze atmospheric conditions and pollution patterns.

### 5. Climate Research
Study long-term fog behavior and climate change impacts.

## Mathematical Background

The analysis uses optimal transport theory to:
- Measure distances between probability distributions on fog data
- Find optimal mappings between different fog configurations
- Quantify transport costs for fog redistribution
- Detect anomalies through OT distance outliers

Key OT concepts used:
- **Wasserstein Distance**: Earth Mover's Distance between fog distributions
- **Transport Plans**: Optimal mappings between fog regions
- **Regularized OT**: Sinkhorn algorithm for smooth transport
- **Barycenter Problems**: Average fog configurations

## Output Examples

- **Spatial Clusters**: Identify 3-5 main fog regions with transport costs
- **Temporal Evolution**: Track fog changes with ~0.001-0.1 Wasserstein distance
- **Quality Scores**: Overall quality in range 0-1 (higher = better)
- **Anomaly Detection**: Typically 5-10% anomalous patterns
- **Interactive Plots**: HTML files for 3D exploration

## Performance Notes

- Large datasets (>100k Gaussians) may require subsampling
- OT computations scale as O(n³) for exact solutions
- Use regularized OT (Sinkhorn) for better scalability
- Parallel processing available for cluster comparisons

## References

- Optimal Transport: "Computational Optimal Transport" by Peyré & Cuturi
- Earth Mover's Distance: Rubner et al. (2000)
- POT Library: https://pythonot.github.io/