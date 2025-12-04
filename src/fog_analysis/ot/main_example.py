#!/usr/bin/env python3
"""
Main example script demonstrating fog optimal transport analysis.
Run this script to perform comprehensive OT analysis on fog data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_ot_analysis import GaussianFogOTAnalysis
from visualization import FogOTVisualization
from applications import FogOTApplications
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main analysis pipeline."""
    print("=== Fog Optimal Transport Analysis ===\n")
    
    # Initialize analyzer
    print("1. Loading fog data...")
    csv_path = "/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/csv/train/dataset_original/truck/truck_fog_beta5_alpha250.csv"
    analyzer = GaussianFogOTAnalysis(csv_path)
    
    # Initialize visualization and applications
    visualizer = FogOTVisualization(analyzer)
    applications = FogOTApplications(analyzer)
    
    print("\n2. Performing feature-based clustering analysis...")
    clusters, cluster_distances, transport_plans, cluster_centers = analyzer.feature_fog_clustering_ot()
    print(f"   - Created {len(np.unique(clusters))} feature clusters")
    print(f"   - Average inter-cluster distance: {np.mean(cluster_distances[cluster_distances > 0]):.4f}")
    
    print("\n3. Analyzing opacity-based transport...")
    opacity_result = analyzer.opacity_based_ot_analysis()
    print(f"   - Wasserstein distance between high/low opacity regions: {opacity_result['wasserstein_distance']:.4f}")
    print(f"   - High opacity gaussians: {len(opacity_result['high_opacity_indices'])}")
    print(f"   - Low opacity gaussians: {len(opacity_result['low_opacity_indices'])}")
    
    print("\n4. Studying temporal fog evolution...")
    time_steps, evolution_distances = analyzer.temporal_fog_evolution_ot()
    if len(evolution_distances) > 0:
        print(f"   - Time steps analyzed: {len(time_steps)}")
        print(f"   - Average evolution distance: {np.mean(evolution_distances):.4f}")
        print(f"   - Evolution trend: {np.polyfit(time_steps, evolution_distances, 1)[0]:.6f}")
    
    print("\n5. Analyzing spatial transport flow...")
    flow_result = analyzer.fog_transport_flow_analysis()
    transport_matrix = flow_result['transport_matrix']
    print(f"   - Transport matrix shape: {transport_matrix.shape}")
    print(f"   - Maximum transport cost: {np.max(transport_matrix):.4f}")
    print(f"   - Minimum non-zero transport cost: {np.min(transport_matrix[transport_matrix > 0]):.4f}")
    
    print("\n6. Multi-feature analysis...")
    multi_result = analyzer.multi_feature_ot_analysis()
    print(f"   - Multi-dimensional Wasserstein distance: {multi_result['wasserstein_distance']:.6f}")
    print(f"   - Feature dimensions: {multi_result['features'].shape[1]}")
    
    print("\n7. Application examples...")
    
    # Fog density estimation
    print("   - Estimating fog density distribution...")
    density_result = applications.fog_density_estimation(grid_resolution=20)
    max_density = np.max(density_result['density'])
    print(f"     Maximum estimated density: {max_density:.6f}")
    
    # Anomaly detection
    print("   - Detecting fog anomalies...")
    anomaly_result = applications.fog_anomaly_detection(contamination=0.05)
    n_anomalies = len(anomaly_result['anomaly_indices'])
    print(f"     Detected {n_anomalies} anomalous fog patterns")
    print(f"     Anomaly threshold: {anomaly_result['threshold']:.4f}")
    
    # Pattern classification
    print("   - Classifying fog patterns...")
    pattern_result = applications.fog_pattern_classification(n_patterns=4)
    print(f"     Converged in {pattern_result['n_iterations']} iterations")
    print(f"     Silhouette score: {pattern_result['silhouette_score']:.4f}")
    
    # Quality assessment
    print("   - Assessing fog quality...")
    quality_result = applications.fog_quality_assessment()
    print(f"     Overall quality score: {quality_result['overall_quality']:.4f}")
    print(f"     Spatial uniformity: {quality_result['spatial_uniformity']:.4f}")
    print(f"     Density consistency: {quality_result['density_consistency']:.4f}")
    print(f"     Temporal stability: {quality_result['temporal_stability']:.4f}")
    print(f"     Feature coherence: {quality_result['feature_coherence']:.4f}")
    
    print("\n8. Generating visualizations...")
    try:
        results = visualizer.save_all_plots("plots")
        print("   - All plots saved successfully!")
        print("   - Check the 'plots' directory for visualization files")
    except Exception as e:
        print(f"   - Warning: Could not save plots - {e}")
    
    print("\n9. Advanced applications...")
    
    # Example: Fog transport optimization
    print("   - Demonstrating transport optimization...")
    positions = analyzer.fog_gaussians['position']
    
    # Define example source and target regions
    center = np.mean(positions, axis=0)
    source_regions = [
        {'type': 'sphere', 'center': center - [1, 0, 0], 'radius': 0.5}
    ]
    target_regions = [
        {'type': 'sphere', 'center': center + [1, 0, 0], 'radius': 0.5}
    ]
    
    transport_opt = applications.fog_transport_optimization(source_regions, target_regions)
    if transport_opt:
        print(f"     Transport cost: {transport_opt['transport_cost']:.4f}")
        print(f"     Source points: {len(transport_opt['source_points'])}")
        print(f"     Target points: {len(transport_opt['target_points'])}")
    
    # Example: Redistribution planning
    print("   - Demonstrating redistribution planning...")
    n_gaussians = len(positions)
    # Create a target distribution (example: uniform)
    target_dist = np.ones(n_gaussians) / n_gaussians
    
    try:
        redist_result = applications.fog_redistribution_planning(target_dist, regularization=0.1)
        print(f"     Total transport cost: {redist_result['total_transport_cost']:.4f}")
        print(f"     Max net change: {np.max(np.abs(redist_result['net_change'])):.6f}")
    except Exception as e:
        print(f"     Warning: Redistribution planning failed - {e}")
    
    print("\n=== Analysis Complete ===")
    print("\nKey Insights:")
    print(f"- Analyzed {len(analyzer.fog_gaussians['position'])} fog Gaussians")
    print(f"- Spatial clustering revealed {len(np.unique(clusters))} distinct regions")
    print(f"- Quality assessment score: {quality_result['overall_quality']:.3f}/1.0")
    print(f"- Detected {n_anomalies} anomalous patterns ({n_anomalies/len(positions)*100:.1f}%)")
    
    return {
        'analyzer': analyzer,
        'visualizer': visualizer,
        'applications': applications,
        'results': {
            'clusters': clusters,
            'opacity_analysis': opacity_result,
            'temporal_evolution': (time_steps, evolution_distances),
            'transport_flow': flow_result,
            'multi_feature': multi_result,
            'quality_assessment': quality_result,
            'anomaly_detection': anomaly_result,
            'pattern_classification': pattern_result
        }
    }

if __name__ == "__main__":
    # Run the analysis
    results = main()
    
    # Optional: Interactive mode
    print("\nTo explore results interactively:")
    print(">>> analyzer = results['analyzer']")
    print(">>> visualizer = results['visualizer']")
    print(">>> applications = results['applications']")
    print(">>> analysis_results = results['results']")