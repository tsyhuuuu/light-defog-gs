import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# ==========================================================
# Utility: Load results into DataFrame
# ==========================================================
def load_results(base_dir, dataset, betas, alphas, conditions):
    """Load metric JSON results from experiment folders."""
    rows = []
    for beta in betas:
        for alpha in alphas:
            folder = os.path.join(base_dir, f"beta{beta}_alpha{alpha}", f"{dataset}/rendered/ours_30000")
            for condition in conditions:
                file_path = os.path.join(folder, f"results_{condition}.json")
                if not os.path.exists(file_path):
                    continue
                with open(file_path, "r") as f:
                    data = json.load(f)
                rows.append({
                    "beta": beta,
                    "alpha": int(alpha),
                    "condition": condition,
                    **data
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid JSON results found — please check your paths.")
    return df

def compute_metrics_stats(base_dir, dataset, betas, alphas, conditions, metrics=("PSNR", "SSIM", "LPIPS")):
    """
    Load JSON results and compute mean and standard deviation for specified metrics.
    
    Returns:
        stats_df: pd.DataFrame with columns ['metric', 'mean', 'std']
    """
    df = load_results(base_dir, dataset, betas, alphas, conditions)
    stats = []
    for condition, group in df.groupby("condition"):
        stats_row = {"condition": condition}
        for metric in metrics:
            if metric in group.columns:
                stats_row[f"{metric}_mean"] = group[metric].mean()
                stats_row[f"{metric}_std"] = group[metric].std()
        stats.append(stats_row)
    
    stats_df = pd.DataFrame(stats)
    
    print("\nMetrics Statistics:")
    print(stats_df)

    return stats_df


def find_best_metrics(df, metrics):
    """Find the best performing condition for each metric."""
    best_conditions = {}
    for metric in metrics:
        # Higher is better for SSIM and PSNR, lower is better for LPIPS
        if metric == "LPIPS":
            best_condition = df.groupby("condition")[metric].mean().idxmin()
        else:
            best_condition = df.groupby("condition")[metric].mean().idxmax()
        best_conditions[metric] = best_condition
    return best_conditions


# ==========================================================
# Interactive 2D plot with alpha slider
# ==========================================================
def create_dynamic_2d_plot(base_dir, dataset, betas, alphas, conditions):
    from matplotlib.widgets import Slider

    metrics = ["SSIM", "PSNR", "LPIPS"]

    # --- Load data ---
    df = load_results(base_dir, dataset, betas, alphas, conditions)
    
    # Find best performing conditions
    best_conditions = find_best_metrics(df, metrics)

    # --- Setup figure ---
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    plt.subplots_adjust(bottom=0.25)
    init_alpha = 25
    lines = {}

    # --- Enhanced color map with highlight colors ---
    base_colors = {"fogged": "#FF6B6B", "defogged": "#4ECDC4", "defogged_clahe": "#95E1D3"}
    highlight_colors = {"fogged": "#FF0000", "defogged": "#0099FF", "defogged_clahe": "#00FF00"}
    
    label_map = {
        "fogged": "Fogged 3DGS",
        "defogged": "Defogged LightGBM",
        "defogged_clahe": "Defogged CLAHE",
    }

    # --- Plot initial lines ---
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        best_condition = best_conditions[metric]
        
        for condition in conditions:
            subset = df[(df["condition"] == condition) & (df["alpha"] == init_alpha)]
            grouped = subset.groupby("beta")[metric].mean()
            
            # Use highlight color for best condition
            is_best = condition == best_condition
            color = highlight_colors[condition] if is_best else base_colors[condition]
            linewidth = 3 if is_best else 2
            marker_size = 10 if is_best else 7
            
            label = f"{label_map[condition]} ★" if is_best else label_map[condition]
            
            line, = ax.plot(
                grouped.index, grouped.values, marker="o",
                color=color, label=label, linewidth=linewidth,
                markersize=marker_size, alpha=0.9 if is_best else 0.7
            )
            lines[(metric, condition)] = line

        ax.set_title(f"{metric} (Best: {label_map[best_condition]})", fontweight='bold')
        ax.set_xlabel("Fog Density β")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    # --- Slider setup ---
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    alpha_slider = Slider(ax_slider, "Brightness A", min(df["alpha"]), max(df["alpha"]),
                          valinit=init_alpha, valstep=25)

    def update(val):
        alpha_val = int(alpha_slider.val)
        for metric in metrics:
            for condition in conditions:
                subset = df[(df["condition"] == condition) & (df["alpha"] == alpha_val)]
                grouped = subset.groupby("beta")[metric].mean()
                betas_full = lines[(metric, condition)].get_xdata()
                yvals = [grouped.get(b, np.nan) for b in betas_full]
                lines[(metric, condition)].set_ydata(yvals)
        alpha_slider.valtext.set_text(f"{alpha_val:03d}")
        fig.canvas.draw_idle()

    alpha_slider.on_changed(update)
    plt.suptitle(f"Metrics Analysis: {dataset}", fontsize=14, fontweight='bold', y=0.98)
    plt.show()


# ==========================================================
# Contour Plot
# ==========================================================
def create_contour_plot(base_dir, dataset, betas, alphas, conditions):
    """Create 2D contour plots for each metric and condition."""
    metrics = ["SSIM", "PSNR", "LPIPS"]
    
    # Readable labels
    condition_labels = {
        "fogged": "Fogged (3DGS)",
        "defogged_clahe": "Defogged (CLAHE)",
        "defogged": "Defogged (3DGS+LightGBM)"
    }
    
    # metric_descriptions = {
    #     "SSIM": "SSIM (Structural Similarity)\nHigher is Better",
    #     "PSNR": "PSNR (Peak Signal-to-Noise Ratio)\nHigher is Better",
    #     "LPIPS": "LPIPS (Perceptual Similarity)\nLower is Better"
    # }

    # --- Load data ---
    df = load_results(base_dir, dataset, betas, alphas, conditions)
    
    # Find best performing conditions
    best_conditions = find_best_metrics(df, metrics)

    # --- Plot contours ---
    fig, axes = plt.subplots(len(conditions), len(metrics), figsize=(15, 4 * len(conditions)))
    if len(conditions) == 1:
        axes = axes.reshape(1, -1)
    
    grid_res = 100
    interp_method = "cubic"

    for row_idx, condition in enumerate(conditions):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            subset = df[df["condition"] == condition]
            X, Y, Z = subset["beta"].values, subset["alpha"].values, subset[metric].values
            
            # Create grid
            xi = np.linspace(X.min(), X.max(), grid_res)
            yi = np.linspace(Y.min(), Y.max(), grid_res)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata((X, Y), Z, (XI, YI), method=interp_method)
            
            # Determine if this is the best condition for this metric
            is_best = condition == best_conditions[metric]
            
            # Create contour plot with appropriate colormap
            if metric == "LPIPS":
                levels = 15
                contour = ax.contourf(XI, YI, ZI, levels=levels, cmap='RdYlGn_r', alpha=0.8)
            else:
                levels = 15
                contour = ax.contourf(XI, YI, ZI, levels=levels, cmap='RdYlGn', alpha=0.8)
            
            # Add contour lines
            contour_lines = ax.contour(XI, YI, ZI, levels=10, colors='black', 
                                      linewidths=0.5, alpha=0.4)
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
            
            # Overlay actual data points
            ax.scatter(X, Y, c='black', s=30, marker='o', alpha=0.6, edgecolors='white', linewidths=0.5)
            
            # Highlight best value
            # if metric == "LPIPS":
            #     best_idx = subset[metric].idxmin()
            # else:
            #     best_idx = subset[metric].idxmax()
            # best_point = subset.loc[best_idx]
            # ax.scatter(best_point["beta"], best_point["alpha"], c='red', s=200, 
            #           marker='*', edgecolors='white', linewidths=2, zorder=5,
            #           label=f'Optimal: {best_point[metric]:.3f}')
            
            # Title with star if best and readable labels
            if is_best:
                ax.set_facecolor('#fffacd')  # Light yellow background for best
                title_text = f"★ {condition_labels[condition]}"
            else:
                title_text = condition_labels[condition]
            
            ax.set_title(title_text, fontsize=12, fontweight='bold' if is_best else 'normal', 
                        pad=10)
            ax.set_xlabel("Fog Density β", fontsize=11, fontweight='bold')
            ax.set_ylabel("Brightness A", fontsize=11, fontweight='bold')
            # ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label(metric, rotation=270, labelpad=20, fontsize=10, fontweight='bold')
            
            # Add metric description as text annotation on the left side
            # if row_idx == 0:
            #     ax.text(0.5, 1.15, metric_descriptions[metric], 
            #            transform=ax.transAxes, fontsize=10, 
            #            ha='center', va='bottom', fontweight='bold',
            #            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

    plt.suptitle(f"Contour Analysis: {dataset}", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


# ==========================================================
# Heatmap Plot
# ==========================================================
def create_heatmap_plot(base_dir, dataset, betas, alphas, conditions):
    """Create heatmap plots for each metric and condition."""
    metrics = ["PSNR", "SSIM", "LPIPS"]
    
    condition_labels = {
        "fogged": "3DGS",            # "Fogged (3DGS)",
        "defogged_clahe": "CLAHE",   # "Defogged (CLAHE)",
        "defogged": "LightDefogGS (Ours)"  # "Defogged (3DGS+LightGBM)"
    }

    # --- Load data ---
    df = load_results(base_dir, dataset, betas, alphas, conditions)

    # --- Plot heatmaps ---
    fig, axes = plt.subplots(len(conditions), len(metrics), figsize=(20, 4 * len(conditions)))
    if len(conditions) == 1:
        axes = axes.reshape(1, -1)

    vmins, vmaxs = [100]*3, [-100]*3
    for row_idx, condition in enumerate(conditions):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            subset = df[df["condition"] == condition]
            
            if subset.empty:
                ax.axis("off")
                continue
            pivot_table = subset.pivot_table(values=metric, index="alpha", 
                                             columns="beta", aggfunc="mean")
            if pivot_table.min().min() < vmins[col_idx]:
                vmins[col_idx] = pivot_table.min().min()
            
            if pivot_table.max().max() > vmaxs[col_idx]:
                vmaxs[col_idx] = pivot_table.max().max()


    for row_idx, condition in enumerate(conditions):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            subset = df[df["condition"] == condition]
            
            if subset.empty:
                ax.axis("off")
                continue

            pivot_table = subset.pivot_table(values=metric, index="alpha", 
                                             columns="beta", aggfunc="mean")

            # Choose colormap
            cmap = 'RdYlBu_r' if metric == "LPIPS" else 'RdYlBu'
            vmin, vmax = vmins[col_idx], vmaxs[col_idx]

            im = ax.imshow(pivot_table.values, cmap=cmap, aspect='auto',
                           origin='lower', vmin=vmin, vmax=vmax, interpolation='bilinear')

            # Tick labels
            ax.set_xticks(range(len(pivot_table.columns)))
            ax.set_xticklabels(pivot_table.columns, fontsize=13.5)
            ax.set_yticks(range(len(pivot_table.index)))
            ax.set_yticklabels(pivot_table.index, fontsize=13.5)

            # --- Find best (alpha, beta) per cell across conditions ---
            best_mask = np.zeros_like(pivot_table.values, dtype=bool)
            for i, alpha in enumerate(pivot_table.index):
                for j, beta in enumerate(pivot_table.columns):
                    all_vals = []
                    for cond in conditions:
                        val = df[(df["condition"] == cond) & 
                                 (df["alpha"] == alpha) & 
                                 (df["beta"] == beta)][metric]
                        if not val.empty:
                            all_vals.append((cond, val.mean()))
                    if not all_vals:
                        continue
                    if metric == "LPIPS":
                        best_cond = min(all_vals, key=lambda x: x[1])[0]  # lower better
                    else:
                        best_cond = max(all_vals, key=lambda x: x[1])[0]  # higher better
                    if condition == best_cond:
                        best_mask[i, j] = True

            # Annotate
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    value = pivot_table.iloc[i, j]
                    if not np.isnan(value):
                        is_best = best_mask[i, j]
                        text_color = "black" if is_best else "#555555"
                        if np.isnan(value):
                            continue
                        ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                                color=text_color, fontsize=11.5,
                                fontweight="bold" if is_best else "normal")

            # Titles and labels
            ax.set_title(condition_labels[condition], fontsize=16.5, fontweight="bold", pad=9)
            ax.set_xlabel("Fog Density β", fontsize=16.5)
            ax.set_ylabel("Brightness A", fontsize=16.5)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric, rotation=270, labelpad=20, fontsize=16.5) # range:5~20


    plt.tight_layout()
    plt.show()




# ==========================================================
# 3D Surface Plot
# ==========================================================
def create_3d_plot(base_dir, dataset, betas, alphas, conditions):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    metrics = ["SSIM", "PSNR", "LPIPS"]

    # --- Load data ---
    df = load_results(base_dir, dataset, betas, alphas, conditions)
    
    # Find best performing conditions
    best_conditions = find_best_metrics(df, metrics)

    # --- Plot surfaces ---
    fig = plt.figure(figsize=(15, 4))
    grid_res = 40
    interp_method = "cubic"

    # Enhanced color scheme
    base_colors = {"fogged": "#FF6B6B", "defogged": "#4ECDC4", "defogged_clahe": "#95E1D3"}
    highlight_colors = {"fogged": "#FF0000", "defogged": "#0099FF", "defogged_clahe": "#00FF00"}

    for idx, metric in enumerate(metrics, 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        best_condition = best_conditions[metric]

        for condition in conditions:
            subset = df[df["condition"] == condition]
            X, Y, Z = subset["beta"], subset["alpha"], subset[metric]
            xi, yi = np.linspace(X.min(), X.max(), grid_res), np.linspace(Y.min(), Y.max(), grid_res)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata((X, Y), Z, (XI, YI), method=interp_method)
            
            # Use highlight color for best condition
            is_best = condition == best_condition
            color = highlight_colors[condition] if is_best else base_colors[condition]
            alpha_val = 0.9 if is_best else 0.5
            
            ax.plot_surface(XI, YI, ZI, color=color, alpha=alpha_val, edgecolor="none")

        best_label = best_condition.replace("_", " ").title()
        ax.set_title(f"{metric}\n★ Best: {best_label}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Fog Density β")
        ax.set_ylabel("Brightness A (α)")
        ax.set_zlabel(metric)
        ax.view_init(elev=20, azim=-30)  # Adjusted: elev=20, azim=-30 makes β more horizontal
        ax.grid(True, alpha=0.3)

    # Enhanced legend
    legend_text = ("★ Highlighted = Best Performance\n\n"
                   "Red = Fogged 3DGS\n"
                   "Cyan = Defogged (LightGBM)\n"
                   "Green = Defogged (CLAHE)")
    fig.text(0.82, 0.85, legend_text, fontsize=10, color="black",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(f"3D Metrics Analysis: {dataset}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==========================================================
# Main with argparse
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3DGS defogging metrics with interactive plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 2D interactive plot with default settings
  python metrics.py --plot-type 2d
  
  # Run 3D plot with custom dataset
  python metrics.py --plot-type 3d --dataset db/custom_scene
  
  # Generate contour plots
  python metrics.py --plot-type contour
  
  # Generate heatmap plots
  python metrics.py --plot-type heatmap
  
  # Generate all plot types
  python metrics.py --plot-type all
  
  # Specify custom beta and alpha ranges
  python metrics.py --plot-type 2d --betas 1 2 3 4 --alpha-step 50
        """
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/tsy/Documents/TeamM_Defog/kpro-dehaze/data/test/tandt_db",
        help="Base directory containing experiment results"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="db/drjohnson",  # tandt/train | tandt/truck | db/playroom | db/drjohnson
        help="Dataset subdirectory name"
    )
    
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["2d", "3d", "contour", "heatmap", "all", "only-stats"],
        default="3d",
        help="Type of plot to generate: 2d (interactive slider), 3d (surface), contour, heatmap, or all"
    )
    
    parser.add_argument(
        "--betas",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="List of beta (fog density) values to analyze"
    )
    
    parser.add_argument(
        "--alpha-min",
        type=int,
        default=0,
        help="Minimum alpha (brightness) value"
    )
    
    parser.add_argument(
        "--alpha-max",
        type=int,
        default=250,
        help="Maximum alpha (brightness) value"
    )
    
    parser.add_argument(
        "--alpha-step",
        type=int,
        default=25,
        help="Step size for alpha values"
    )
    
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["fogged", "defogged_clahe", "defogged"],
        help="List of conditions to compare"
    )
    
    args = parser.parse_args()
    
    # Generate alpha values
    alphas = [f"{i:03d}" for i in range(args.alpha_min, args.alpha_max + 1, args.alpha_step)]
    
    print(f"Configuration:")
    print(f"  Base directory: {args.base_dir}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Plot type: {args.plot_type}")
    print(f"  Betas: {args.betas}")
    print(f"  Alphas: {args.alpha_min} to {args.alpha_max} (step {args.alpha_step})")
    print(f"  Conditions: {args.conditions}")
    print()
    
    if args.plot_type in ["2d", "all"]:
        print("Generating 2D interactive plot...")
        create_dynamic_2d_plot(args.base_dir, args.dataset, args.betas, alphas, args.conditions)
    
    if args.plot_type in ["3d", "all"]:
        print("Generating 3D surface plot...")
        create_3d_plot(args.base_dir, args.dataset, args.betas, alphas, args.conditions)
    
    if args.plot_type in ["contour", "all"]:
        print("Generating contour plot...")
        create_contour_plot(args.base_dir, args.dataset, args.betas, alphas, args.conditions)
    
    if args.plot_type in ["heatmap", "all"]:
        print("Generating heatmap plot...")
        create_heatmap_plot(args.base_dir, args.dataset, args.betas, alphas, args.conditions)
    if args.plot_type in ["only-stats"]:
        print("Computing and displaying metrics statistics...")
        compute_metrics_stats(args.base_dir, args.dataset, args.betas, alphas, args.conditions, metrics=("PSNR", "SSIM", "LPIPS"))



if __name__ == "__main__":
    main()