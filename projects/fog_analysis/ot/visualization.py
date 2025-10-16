import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class FogOTVisualization:
    """
    Visualization tools for fog optimal transport analysis.
    """
    
    def __init__(self, ot_analyzer):
        """Initialize with OT analyzer."""
        self.analyzer = ot_analyzer
        
    def plot_spatial_clusters(self, clusters, cluster_centers, cluster_distances):
        """Visualize spatial fog clustering with OT distances."""
        fig = plt.figure(figsize=(15, 5))
        
        # 3D scatter plot of clusters
        ax1 = fig.add_subplot(131, projection='3d')
        positions = self.analyzer.fog_gaussians['position']
        
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                            c=clusters, cmap='viridis', alpha=0.6)
        ax1.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                   c='red', marker='x', s=200, linewidths=3)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')
        ax1.set_title('3D Fog Clusters')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)
        
        # 2D projection
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, linewidths=3)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('2D Projection (XY)')
        plt.colorbar(scatter2, ax=ax2)
        
        # Cluster distance heatmap
        ax3 = fig.add_subplot(133)
        sns.heatmap(cluster_distances, annot=True, cmap='viridis', ax=ax3)
        ax3.set_title('Wasserstein Distances Between Clusters')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Cluster')
        
        plt.tight_layout()
        return fig
    
    def plot_opacity_analysis(self, opacity_result):
        """Visualize opacity-based OT analysis."""
        fig = plt.figure(figsize=(15, 5))
        
        # High vs low opacity positions
        ax1 = fig.add_subplot(131, projection='3d')
        high_pos = opacity_result['high_opacity_positions']
        low_pos = opacity_result['low_opacity_positions']
        
        ax1.scatter(high_pos[:, 0], high_pos[:, 1], high_pos[:, 2], 
                   c='red', alpha=0.6, label='High Opacity', s=20)
        ax1.scatter(low_pos[:, 0], low_pos[:, 1], low_pos[:, 2], 
                   c='blue', alpha=0.6, label='Low Opacity', s=20)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')
        ax1.set_title('High vs Low Opacity Regions')
        ax1.legend()
        
        # Opacity distribution
        ax2 = fig.add_subplot(132)
        ax2.hist(opacity_result['high_opacity_values'], alpha=0.7, label='High Opacity', bins=30, color='red')
        ax2.hist(opacity_result['low_opacity_values'], alpha=0.7, label='Low Opacity', bins=30, color='blue')
        ax2.set_xlabel('Opacity Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Opacity Distribution')
        ax2.legend()
        
        # Transport plan visualization (sample)
        ax3 = fig.add_subplot(133)
        transport_plan = opacity_result['transport_plan']
        im = ax3.imshow(transport_plan[:100, :100], cmap='viridis', aspect='auto')
        ax3.set_title('Transport Plan (Sample 100x100)')
        ax3.set_xlabel('Low Opacity Gaussians')
        ax3.set_ylabel('High Opacity Gaussians')
        plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_evolution(self, time_steps, evolution_distances):
        """Plot fog evolution over time using OT distances."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(time_steps, evolution_distances, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Fog Evolution: Wasserstein Distance Between Consecutive Time Steps')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(time_steps) > 1:
            z = np.polyfit(time_steps, evolution_distances, 1)
            p = np.poly1d(z)
            ax.plot(time_steps, p(time_steps), "r--", alpha=0.8, label=f'Trend: slope={z[0]:.4f}')
            ax.legend()
        
        return fig
    
    def plot_transport_flow(self, flow_result):
        """Visualize fog transport flow between spatial regions."""
        fig = plt.figure(figsize=(15, 10))
        
        # 2D spatial distribution with quadrants
        ax1 = fig.add_subplot(221)
        pos_2d = flow_result['pos_2d']
        quadrants = flow_result['quadrants']
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (quad, color, name) in enumerate(zip(quadrants, colors, flow_result['quadrant_names'])):
            if len(quad) > 0:
                ax1.scatter(pos_2d[quad, 0], pos_2d[quad, 1], c=color, alpha=0.6, label=name, s=20)
        
        ax1.axhline(y=np.median(pos_2d[:, 1]), color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=np.median(pos_2d[:, 0]), color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('Spatial Quadrants (PCA Space)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Transport matrix heatmap
        ax2 = fig.add_subplot(222)
        transport_matrix = flow_result['transport_matrix']
        im = sns.heatmap(transport_matrix, annot=True, cmap='viridis', 
                        xticklabels=flow_result['quadrant_names'],
                        yticklabels=flow_result['quadrant_names'], ax=ax2)
        ax2.set_title('Transport Cost Matrix Between Quadrants')
        
        # Flow diagram
        ax3 = fig.add_subplot(223)
        # Create a simplified flow diagram
        quad_centers = []
        for quad in quadrants:
            if len(quad) > 0:
                center = np.mean(pos_2d[quad], axis=0)
                quad_centers.append(center)
            else:
                quad_centers.append([0, 0])
        
        quad_centers = np.array(quad_centers)
        
        # Plot quadrant centers
        for i, (center, color, name) in enumerate(zip(quad_centers, colors, flow_result['quadrant_names'])):
            ax3.scatter(center[0], center[1], c=color, s=200, alpha=0.8, label=name)
            ax3.annotate(f'Q{i+1}', (center[0], center[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Draw flow arrows based on transport costs
        for i in range(4):
            for j in range(4):
                if i != j and transport_matrix[i, j] > 0:
                    # Arrow thickness proportional to inverse of transport cost
                    arrow_width = max(0.1, 2.0 / (transport_matrix[i, j] + 0.1))
                    ax3.annotate('', xy=quad_centers[j], xytext=quad_centers[i],
                               arrowprops=dict(arrowstyle='->', lw=arrow_width, alpha=0.6, color='gray'))
        
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title('Transport Flow Between Quadrants')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Transport cost distribution
        ax4 = fig.add_subplot(224)
        costs = transport_matrix[transport_matrix > 0]
        ax4.hist(costs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Transport Cost')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Transport Costs')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multi_feature_analysis(self, multi_result):
        """Visualize multi-feature OT analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature space visualization (first 3 dimensions)
        ax1 = axes[0, 0]
        features = multi_result['features']
        subset1_idx = multi_result['subset1_idx']
        subset2_idx = multi_result['subset2_idx']
        
        ax1.scatter(features[subset1_idx, 0], features[subset1_idx, 1], 
                   alpha=0.6, label='Subset 1', s=20)
        ax1.scatter(features[subset2_idx, 0], features[subset2_idx, 1], 
                   alpha=0.6, label='Subset 2', s=20)
        ax1.set_xlabel('Normalized Position X')
        ax1.set_ylabel('Normalized Position Y')
        ax1.set_title('Multi-Feature Space (Position Components)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Transport plan heatmap (sample)
        ax2 = axes[0, 1]
        transport_plan = multi_result['transport_plan']
        sample_size = min(100, transport_plan.shape[0], transport_plan.shape[1])
        im = ax2.imshow(transport_plan[:sample_size, :sample_size], cmap='viridis', aspect='auto')
        ax2.set_title(f'Transport Plan (Sample {sample_size}x{sample_size})')
        ax2.set_xlabel('Subset 2')
        ax2.set_ylabel('Subset 1')
        plt.colorbar(im, ax=ax2)
        
        # Feature importance (variance)
        ax3 = axes[1, 0]
        feature_vars = np.var(features, axis=0)
        feature_names = ['Pos_X', 'Pos_Y', 'Pos_Z', 'Opacity', 'Scale']
        ax3.bar(feature_names, feature_vars)
        ax3.set_ylabel('Variance')
        ax3.set_title('Feature Importance (Variance)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Wasserstein distance info
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, f"Wasserstein Distance: {multi_result['wasserstein_distance']:.6f}", 
                fontsize=14, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f"Subset 1 size: {len(subset1_idx)}", 
                fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f"Subset 2 size: {len(subset2_idx)}", 
                fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.2, f"Feature dimensions: {features.shape[1]}", 
                fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Analysis Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_3d_plot(self):
        """Create interactive 3D plot using plotly."""
        positions = self.analyzer.fog_gaussians['position']
        opacities = self.analyzer.fog_gaussians['opacity']
        
        fig = go.Figure(data=[go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=opacities,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Opacity")
            ),
            text=[f'Opacity: {op:.3f}' for op in opacities],
            hovertemplate='<b>Position</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<br>' +
                         '%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Interactive 3D Fog Gaussian Distribution",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position", 
                zaxis_title="Z Position"
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def save_all_plots(self, output_dir="fog_analysis/ot/plots"):
        """Generate and save all analysis plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating spatial clustering analysis...")
        clusters, cluster_distances, _, cluster_centers = self.analyzer.feature_fog_clustering_ot()
        fig1 = self.plot_spatial_clusters(clusters, cluster_centers, cluster_distances)
        fig1.savefig(f"{output_dir}/spatial_clusters.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        print("Generating opacity analysis...")
        opacity_result = self.analyzer.opacity_based_ot_analysis()
        fig2 = self.plot_opacity_analysis(opacity_result)
        fig2.savefig(f"{output_dir}/opacity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print("Generating temporal evolution analysis...")
        time_steps, evolution_distances = self.analyzer.temporal_fog_evolution_ot()
        fig3 = self.plot_temporal_evolution(time_steps, evolution_distances)
        fig3.savefig(f"{output_dir}/temporal_evolution.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        print("Generating transport flow analysis...")
        flow_result = self.analyzer.fog_transport_flow_analysis()
        fig4 = self.plot_transport_flow(flow_result)
        fig4.savefig(f"{output_dir}/transport_flow.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        print("Generating multi-feature analysis...")
        multi_result = self.analyzer.multi_feature_ot_analysis()
        fig5 = self.plot_multi_feature_analysis(multi_result)
        fig5.savefig(f"{output_dir}/multi_feature_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        print("Creating interactive 3D plot...")
        fig_interactive = self.create_interactive_3d_plot()
        fig_interactive.write_html(f"{output_dir}/interactive_3d_fog.html")
        
        print(f"All plots saved to {output_dir}/")
        
        return {
            'spatial_clusters': cluster_distances,
            'opacity_analysis': opacity_result,
            'temporal_evolution': (time_steps, evolution_distances),
            'transport_flow': flow_result,
            'multi_feature': multi_result
        }