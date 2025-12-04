import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import ot

class FogOTApplications:
    """
    Practical applications of optimal transport for fog analysis.
    """
    
    def __init__(self, ot_analyzer):
        """Initialize with OT analyzer."""
        self.analyzer = ot_analyzer
    
    def fog_density_estimation(self, grid_resolution=50):
        """
        Estimate fog density distribution using optimal transport.
        """
        positions = self.analyzer.fog_gaussians['position']
        opacities = self.analyzer.fog_gaussians['opacity']
        
        # Create 3D grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        z_grid = np.linspace(z_min, z_max, grid_resolution)
        
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Use OT to transport fog mass to grid points
        fog_weights = opacities / np.sum(opacities)
        grid_weights = np.ones(len(grid_points)) / len(grid_points)
        
        # Compute transport cost
        M = ot.dist(positions, grid_points, metric='euclidean')
        
        # Solve optimal transport
        transport_plan = ot.emd(fog_weights, grid_weights, M)
        
        # Estimate density at each grid point
        density = np.sum(transport_plan, axis=0)
        density = density.reshape(X.shape)
        
        return {
            'grid_points': grid_points,
            'density': density,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'z_grid': z_grid,
            'X': X, 'Y': Y, 'Z': Z
        }
    
    def fog_anomaly_detection(self, contamination=0.1):
        """
        Detect anomalous fog patterns using OT-based distances.
        """
        positions = self.analyzer.fog_gaussians['position']
        opacities = self.analyzer.fog_gaussians['opacity']
        scales = self.analyzer.fog_gaussians['scale']
        
        # Combine features
        features = np.column_stack([
            positions,
            opacities.reshape(-1, 1),
            np.linalg.norm(scales, axis=1).reshape(-1, 1)
        ])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        n_samples = len(features_scaled)
        ot_distances = np.zeros(n_samples)
        
        # For each sample, compute OT distance to rest of the dataset
        for i in range(n_samples):
            # Current sample
            sample = features_scaled[i:i+1]
            
            # Rest of the dataset
            rest_idx = np.concatenate([np.arange(i), np.arange(i+1, n_samples)])
            rest_samples = features_scaled[rest_idx]
            
            if len(rest_samples) > 0:
                # Uniform weights
                a = np.array([1.0])
                b = np.ones(len(rest_samples)) / len(rest_samples)
                
                # Cost matrix
                M = ot.dist(sample, rest_samples, metric='euclidean')
                
                # OT distance
                ot_distances[i] = ot.emd2(a, b, M)
        
        # Identify anomalies (high OT distances)
        threshold = np.percentile(ot_distances, (1 - contamination) * 100)
        anomalies = ot_distances > threshold
        
        return {
            'ot_distances': ot_distances,
            'anomalies': anomalies,
            'threshold': threshold,
            'anomaly_indices': np.where(anomalies)[0],
            'anomaly_positions': positions[anomalies],
            'anomaly_scores': ot_distances[anomalies]
        }
    
    def fog_pattern_classification(self, n_patterns=5):
        """
        Classify fog patterns using OT-based clustering.
        """
        positions = self.analyzer.fog_gaussians['position']
        opacities = self.analyzer.fog_gaussians['opacity']
        
        # Create feature matrix
        features = np.column_stack([positions, opacities])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Initialize pattern centers randomly
        n_samples = len(features_scaled)
        pattern_centers = features_scaled[np.random.choice(n_samples, n_patterns, replace=False)]
        
        # OT-based k-means clustering
        max_iter = 50
        tolerance = 1e-6
        
        for iteration in range(max_iter):
            # Assign samples to nearest pattern center (OT distance)
            assignments = np.zeros(n_samples, dtype=int)
            
            for i in range(n_samples):
                sample = features_scaled[i:i+1]
                min_distance = float('inf')
                best_pattern = 0
                
                for j in range(n_patterns):
                    center = pattern_centers[j:j+1]
                    
                    # OT distance between sample and center
                    a = np.array([1.0])
                    b = np.array([1.0])
                    M = ot.dist(sample, center, metric='euclidean')
                    distance = ot.emd2(a, b, M)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_pattern = j
                
                assignments[i] = best_pattern
            
            # Update pattern centers
            new_centers = np.zeros_like(pattern_centers)
            for j in range(n_patterns):
                pattern_samples = features_scaled[assignments == j]
                if len(pattern_samples) > 0:
                    new_centers[j] = np.mean(pattern_samples, axis=0)
                else:
                    new_centers[j] = pattern_centers[j]
            
            # Check convergence
            center_shift = np.linalg.norm(new_centers - pattern_centers)
            pattern_centers = new_centers
            
            if center_shift < tolerance:
                break
        
        # Compute silhouette score
        if len(np.unique(assignments)) > 1:
            silhouette = silhouette_score(features_scaled, assignments)
        else:
            silhouette = -1
        
        return {
            'assignments': assignments,
            'pattern_centers': pattern_centers,
            'n_iterations': iteration + 1,
            'silhouette_score': silhouette,
            'features_scaled': features_scaled,
            'scaler': scaler
        }
    
    def fog_transport_optimization(self, source_regions, target_regions):
        """
        Optimize fog transport between specified regions.
        """
        positions = self.analyzer.fog_gaussians['position']
        opacities = self.analyzer.fog_gaussians['opacity']
        
        # Extract source and target points
        source_points = []
        source_weights = []
        target_points = []
        target_weights = []
        
        for region in source_regions:
            mask = self._create_region_mask(positions, region)
            if np.any(mask):
                source_points.append(positions[mask])
                source_weights.append(opacities[mask])
        
        for region in target_regions:
            mask = self._create_region_mask(positions, region)
            if np.any(mask):
                target_points.append(positions[mask])
                target_weights.append(np.ones(np.sum(mask)))
        
        if not source_points or not target_points:
            return None
        
        # Combine all source and target points
        all_source_points = np.vstack(source_points)
        all_source_weights = np.concatenate(source_weights)
        all_target_points = np.vstack(target_points)
        all_target_weights = np.concatenate(target_weights)
        
        # Normalize weights
        all_source_weights = all_source_weights / np.sum(all_source_weights)
        all_target_weights = all_target_weights / np.sum(all_target_weights)
        
        # Compute optimal transport
        M = ot.dist(all_source_points, all_target_points, metric='euclidean')
        transport_plan = ot.emd(all_source_weights, all_target_weights, M)
        transport_cost = ot.emd2(all_source_weights, all_target_weights, M)
        
        return {
            'source_points': all_source_points,
            'target_points': all_target_points,
            'source_weights': all_source_weights,
            'target_weights': all_target_weights,
            'transport_plan': transport_plan,
            'transport_cost': transport_cost,
            'cost_matrix': M
        }
    
    def _create_region_mask(self, positions, region):
        """
        Create boolean mask for points within a specified region.
        Region format: {'type': 'sphere', 'center': [x, y, z], 'radius': r}
                      or {'type': 'box', 'min': [x_min, y_min, z_min], 'max': [x_max, y_max, z_max]}
        """
        if region['type'] == 'sphere':
            center = np.array(region['center'])
            radius = region['radius']
            distances = np.linalg.norm(positions - center, axis=1)
            return distances <= radius
        
        elif region['type'] == 'box':
            min_coords = np.array(region['min'])
            max_coords = np.array(region['max'])
            return np.all((positions >= min_coords) & (positions <= max_coords), axis=1)
        
        else:
            raise ValueError(f"Unknown region type: {region['type']}")
    
    def fog_redistribution_planning(self, target_distribution, regularization=0.01):
        """
        Plan optimal fog redistribution to achieve target distribution.
        """
        positions = self.analyzer.fog_gaussians['position']
        current_opacities = self.analyzer.fog_gaussians['opacity']
        
        # Current distribution (normalized)
        current_weights = current_opacities / np.sum(current_opacities)
        
        # Target distribution (should sum to 1)
        if len(target_distribution) != len(positions):
            raise ValueError("Target distribution must have same length as number of fog gaussians")
        
        target_weights = np.array(target_distribution)
        target_weights = target_weights / np.sum(target_weights)
        
        # Compute transport plan with regularization (entropic OT)
        M = np.zeros((len(positions), len(positions)))  # Self-transport
        for i in range(len(positions)):
            for j in range(len(positions)):
                M[i, j] = np.linalg.norm(positions[i] - positions[j])
        
        # Regularized optimal transport
        transport_plan = ot.sinkhorn(current_weights, target_weights, M, regularization)
        
        # Compute redistribution instructions
        redistribution_matrix = transport_plan
        net_change = np.sum(redistribution_matrix, axis=0) - np.sum(redistribution_matrix, axis=1)
        
        return {
            'current_distribution': current_weights,
            'target_distribution': target_weights,
            'transport_plan': transport_plan,
            'redistribution_matrix': redistribution_matrix,
            'net_change': net_change,
            'total_transport_cost': np.sum(transport_plan * M),
            'positions': positions
        }
    
    def fog_quality_assessment(self):
        """
        Assess fog quality based on OT analysis metrics.
        """
        # Spatial uniformity
        clusters, cluster_distances, _, _ = self.analyzer.feature_fog_clustering_ot()
        spatial_uniformity = 1.0 / (1.0 + np.std(cluster_distances[cluster_distances > 0]))
        
        # Density consistency
        opacity_result = self.analyzer.opacity_based_ot_analysis()
        density_consistency = 1.0 / (1.0 + opacity_result['wasserstein_distance'])
        
        # Temporal stability
        time_steps, evolution_distances = self.analyzer.temporal_fog_evolution_ot()
        if len(evolution_distances) > 0:
            temporal_stability = 1.0 / (1.0 + np.std(evolution_distances))
        else:
            temporal_stability = 1.0
        
        # Multi-feature coherence
        multi_result = self.analyzer.multi_feature_ot_analysis()
        feature_coherence = 1.0 / (1.0 + multi_result['wasserstein_distance'])
        
        # Overall quality score (weighted average)
        weights = [0.3, 0.3, 0.2, 0.2]  # Adjustable weights
        quality_score = (weights[0] * spatial_uniformity + 
                        weights[1] * density_consistency +
                        weights[2] * temporal_stability +
                        weights[3] * feature_coherence)
        
        return {
            'spatial_uniformity': spatial_uniformity,
            'density_consistency': density_consistency,
            'temporal_stability': temporal_stability,
            'feature_coherence': feature_coherence,
            'overall_quality': quality_score,
            'quality_breakdown': {
                'spatial_uniformity': spatial_uniformity,
                'density_consistency': density_consistency,
                'temporal_stability': temporal_stability,
                'feature_coherence': feature_coherence
            }
        }