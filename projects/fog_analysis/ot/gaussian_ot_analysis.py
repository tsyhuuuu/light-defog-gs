import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

class GaussianFogOTAnalysis:
    """
    Optimal Transport analysis for Gaussian fog/haze data.
    """
    
    def __init__(self, csv_path):
        """Initialize with fog data."""
        self.data = pd.read_csv(csv_path)
        self.gaussians = None
        self.fog_gaussians = None
        self.prepare_gaussian_data()
    
    def prepare_gaussian_data(self):
        """Extract Gaussian parameters from the dataset."""
        # Extract position, scale, and opacity for each Gaussian
        self.gaussians = {
            'position': self.data[['pos_x', 'pos_y', 'pos_z']].values,
            'scale': self.data[['scale_x', 'scale_y', 'scale_z']].values,
            'rotation': self.data[['rot_0', 'rot_1', 'rot_2', 'rot_3']].values,
            'opacity': self.data['opacity'].values,
            'color': self.data[['f_dc_0', 'f_dc_1', 'f_dc_2']].values,
            'beta': self.data['beta'].values,
            'alpha': self.data['alpha'].values,
            'is_fog': self.data['is_fog'].values,
            'id': self.data['id'].values
        }
        
        # Filter only fog gaussians
        fog_mask = self.gaussians['is_fog'] == 1.0
        self.fog_gaussians = {key: val[fog_mask] for key, val in self.gaussians.items()}
        
        print(f"Total Gaussians: {len(self.gaussians['position'])}")
        print(f"Fog Gaussians: {len(self.fog_gaussians['position'])}")
    
    def compute_gaussian_wasserstein_distance(self, subset1_idx, subset2_idx, feature='opacity'):
        """
        Compute Wasserstein distance between two subsets of Gaussians.
        """
        X = self.fog_gaussians[feature][subset1_idx]
        Y = self.fog_gaussians[feature][subset2_idx]
        
        # Equal weights for all points
        a = np.ones(len(X)) / len(X)
        b = np.ones(len(Y)) / len(Y)
        
        # Compute cost matrix (Euclidean distance)
        M = cdist(X, Y, metric='euclidean')
        
        # Solve optimal transport
        W = ot.emd2(a, b, M)
        return W, ot.emd(a, b, M)
    
    def feature_fog_clustering_ot(self, n_clusters=5):
        """
        Use optimal transport to analyze fog distribution based on opacity, scale, and color features.
        """
        # Combine opacity, scale magnitude, and colors as features
        opacities = self.fog_gaussians['opacity'].reshape(-1, 1)
        scales = np.linalg.norm(self.fog_gaussians['scale'], axis=1).reshape(-1, 1)
        colors = self.fog_gaussians['color']
        
        features = np.hstack([
            opacities / np.std(opacities),
            scales / np.std(scales),
            colors / np.std(colors, axis=0)
        ])
        
        # K-means clustering for initial grouping
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Compute OT distances between clusters
        cluster_distances = np.zeros((n_clusters, n_clusters))
        transport_plans = {}
        
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                idx_i = np.where(clusters == i)[0]
                idx_j = np.where(clusters == j)[0]
                
                if len(idx_i) > 0 and len(idx_j) > 0:
                    # Use combined features for distance computation
                    X = features[idx_i]
                    Y = features[idx_j]
                    
                    a = np.ones(len(X)) / len(X)
                    b = np.ones(len(Y)) / len(Y)
                    M = cdist(X, Y, metric='euclidean')
                    
                    dist = ot.emd2(a, b, M)
                    plan = ot.emd(a, b, M)
                    
                    cluster_distances[i, j] = dist
                    cluster_distances[j, i] = dist
                    transport_plans[(i, j)] = plan
        
        return clusters, cluster_distances, transport_plans, kmeans.cluster_centers_
    
    def opacity_based_ot_analysis(self):
        """
        Analyze fog density distribution using optimal transport on opacity values.
        """
        opacities = self.fog_gaussians['opacity']
        positions = self.fog_gaussians['position']
        
        # Divide into high and low opacity regions
        median_opacity = np.median(opacities)
        high_opacity_idx = np.where(opacities >= median_opacity)[0]
        low_opacity_idx = np.where(opacities < median_opacity)[0]
        
        # Compute OT between high and low opacity regions using opacity values
        X = opacities[high_opacity_idx].reshape(-1, 1)
        Y = opacities[low_opacity_idx].reshape(-1, 1)
        
        a = np.ones(len(X)) / len(X)
        b = np.ones(len(Y)) / len(Y)
        M = cdist(X, Y, metric='euclidean')
        
        W_dist = ot.emd2(a, b, M)
        transport_plan = ot.emd(a, b, M)
        
        return {
            'wasserstein_distance': W_dist,
            'transport_plan': transport_plan,
            'high_opacity_values': opacities[high_opacity_idx],
            'low_opacity_values': opacities[low_opacity_idx],
            'high_opacity_indices': high_opacity_idx,
            'low_opacity_indices': low_opacity_idx
        }
    
    def temporal_fog_evolution_ot(self, n_time_steps=10):
        """
        Simulate temporal fog evolution using optimal transport on feature space.
        Assumes fog IDs represent temporal ordering.
        """
        ids = self.fog_gaussians['id']
        opacities = self.fog_gaussians['opacity']
        scales = self.fog_gaussians['scale']
        colors = self.fog_gaussians['color']
        
        # Create feature space
        features = np.hstack([
            opacities.reshape(-1, 1) / np.std(opacities),
            np.linalg.norm(scales, axis=1).reshape(-1, 1) / np.std(np.linalg.norm(scales, axis=1)),
            colors / np.std(colors, axis=0)
        ])
        
        # Sort by ID to create temporal ordering
        sorted_idx = np.argsort(ids)
        n_gaussians = len(sorted_idx)
        
        # Create time steps
        step_size = n_gaussians // n_time_steps
        time_steps = []
        evolution_distances = []
        
        for t in range(n_time_steps - 1):
            start_idx = t * step_size
            end_idx = (t + 1) * step_size
            next_start = (t + 1) * step_size
            next_end = (t + 2) * step_size
            
            if next_end > n_gaussians:
                next_end = n_gaussians
            
            current_step = sorted_idx[start_idx:end_idx]
            next_step = sorted_idx[next_start:next_end]
            
            if len(current_step) > 0 and len(next_step) > 0:
                X = features[current_step]
                Y = features[next_step]
                
                a = np.ones(len(X)) / len(X)
                b = np.ones(len(Y)) / len(Y)
                M = cdist(X, Y, metric='euclidean')
                
                W_dist = ot.emd2(a, b, M)
                evolution_distances.append(W_dist)
                time_steps.append(t)
        
        return time_steps, evolution_distances
    
    def fog_transport_flow_analysis(self):
        """
        Analyze fog transport flow patterns using optimal transport.
        """
        positions = self.fog_gaussians['position']
        opacities = self.fog_gaussians['opacity']
        
        # Create source and sink regions based on spatial distribution
        # Source: regions with high fog density
        # Sink: regions with low fog density
        
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(positions)
        
        # Divide space into source and sink regions
        x_median = np.median(pos_2d[:, 0])
        y_median = np.median(pos_2d[:, 1])
        
        # Quadrant-based analysis
        q1 = np.where((pos_2d[:, 0] >= x_median) & (pos_2d[:, 1] >= y_median))[0]
        q2 = np.where((pos_2d[:, 0] < x_median) & (pos_2d[:, 1] >= y_median))[0]
        q3 = np.where((pos_2d[:, 0] < x_median) & (pos_2d[:, 1] < y_median))[0]
        q4 = np.where((pos_2d[:, 0] >= x_median) & (pos_2d[:, 1] < y_median))[0]
        
        quadrants = [q1, q2, q3, q4]
        quadrant_names = ['Q1 (NE)', 'Q2 (NW)', 'Q3 (SW)', 'Q4 (SE)']
        
        # Compute transport between all quadrant pairs
        transport_matrix = np.zeros((4, 4))
        
        for i in range(4):
            for j in range(4):
                if i != j and len(quadrants[i]) > 0 and len(quadrants[j]) > 0:
                    W_dist, _ = self.compute_gaussian_wasserstein_distance(
                        quadrants[i], quadrants[j], 'position'
                    )
                    transport_matrix[i, j] = W_dist
        
        return {
            'transport_matrix': transport_matrix,
            'quadrant_names': quadrant_names,
            'quadrants': quadrants,
            'pos_2d': pos_2d,
            'pca': pca
        }
    
    def multi_feature_ot_analysis(self):
        """
        Multi-dimensional optimal transport analysis considering opacity, scale, and color only.
        """
        # Normalize features (excluding position and rotation)
        opacities = self.fog_gaussians['opacity'].reshape(-1, 1)
        scales = np.linalg.norm(self.fog_gaussians['scale'], axis=1).reshape(-1, 1)
        colors = self.fog_gaussians['color']
        
        # Combine features
        features = np.hstack([
            opacities / np.std(opacities),
            scales / np.std(scales),
            colors / np.std(colors, axis=0)
        ])
        
        # Divide into two random subsets for comparison
        n_total = len(features)
        idx = np.random.permutation(n_total)
        subset1_idx = idx[:n_total//2]
        subset2_idx = idx[n_total//2:]
        
        X = features[subset1_idx]
        Y = features[subset2_idx]
        
        # Equal weights
        a = np.ones(len(X)) / len(X)
        b = np.ones(len(Y)) / len(Y)
        
        # Compute cost matrix
        M = cdist(X, Y, metric='euclidean')
        
        # Solve optimal transport
        W_dist = ot.emd2(a, b, M)
        transport_plan = ot.emd(a, b, M)
        
        return {
            'wasserstein_distance': W_dist,
            'transport_plan': transport_plan,
            'features': features,
            'subset1_idx': subset1_idx,
            'subset2_idx': subset2_idx
        }