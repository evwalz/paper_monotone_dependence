#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import transforms
from helper import get_acc, get_rmse, calculate_improvement_over_climatology, get_seeps, calculate_improvement_over_climatology_corr, get_cma

def label_panel(ax, letter, *, offset_left=0.0, offset_up=0.08, prefix='', postfix=')', fs=16, **font_kwds):
    """Add panel labels (a, b, c, d) to subplots."""
    kwds = dict(fontsize=fs)
    kwds.update(font_kwds)
    fig = ax.figure
    trans = ax.transAxes + transforms.ScaledTranslation(-offset_left, offset_up, fig.dpi_scale_trans)
    ax.text(0, 1, prefix + letter + postfix, transform=trans, **kwds)

def setup_plot_style():
    """Configure global matplotlib settings."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,     
        'axes.labelsize': 14,     
        'xtick.labelsize': 12,    
        'ytick.labelsize': 12,    
        'legend.fontsize': 12,    
    })

def get_color_mapping():
    """Define color mapping for different models."""
    color_codes = sns.color_palette("colorblind", 6)
    return {
        'Climatology': color_codes[2],
        'Persistence': '#9467bd',
        'GraphCast': color_codes[0],
        'IFS HRES': color_codes[1]
    }

def setup_latitude_axis(ax):
    """Configure latitude axis with proper tick labels."""
    ax.set_xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
    ax.set_xticklabels(['80°S', '60°S', '40°S', '20°S', 'Equator', '20°N', '40°N', '60°N', '80°N'])
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax.grid(True)

def plot_metric(ax, data, latitudes, color_map, ylabel, panel_letter, linewidth=2, legend = False):
    """Plot a single metric across latitudes."""
    for model_name, values in data.items():
        label = 'HRES' if model_name == 'IFS HRES' else model_name
        ax.plot(latitudes, values, label=label, color=color_map[model_name], linewidth=linewidth)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Latitude (degrees)')
    setup_latitude_axis(ax)
    if legend:
        ax.legend()
    label_panel(ax, panel_letter)

def load_data(data_results_dir):
    """Load all performance metrics data."""
    print("Loading data...")
    
    # Load RMSE data and calculate improvements
    rmse_per_lats = get_rmse(data_results_dir + 'rmse_results/')
    rmse_improvement = calculate_improvement_over_climatology(rmse_per_lats)
    
    # Load ACC data
    acc_per_lat = get_acc(data_results_dir + 'acc_results/')
    
    # Load SEEPS data and calculate improvements
    seeps_per_lats = get_seeps(data_results_dir + 'seeps_results/')
    seeps_improvement = calculate_improvement_over_climatology(seeps_per_lats)
    
    # Load CMA data and calculate improvements
    cma_data = get_cma(data_results_dir + '/cma_results/')
    cma_improvement = calculate_improvement_over_climatology_corr(cma_data)
    
    return rmse_improvement, acc_per_lat, seeps_improvement, cma_improvement

def create_performance_plots(data_results_dir, output_dir, filename='combined_model_performance_2x2.pdf'):
    """Create and save the 2x2 performance comparison plot."""
    
    # Setup
    setup_plot_style()
    color_map = get_color_mapping()
    latitudes = np.linspace(-90, 90, 121)
    
    # Load data
    rmse_improvement, acc_per_lat, seeps_improvement, cma_improvement = load_data(data_results_dir)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot each metric
    plot_metric(axes[0, 0], rmse_improvement, latitudes, color_map, 
                'RMSE Improvement (%)', 'a', legend = True)
    
    plot_metric(axes[0, 1], acc_per_lat, latitudes, color_map, 
                'ACC', 'b')
    
    plot_metric(axes[1, 0], seeps_improvement, latitudes, color_map, 
                'SEEPS Improvement (%)', 'c')
    
    plot_metric(axes[1, 1], cma_improvement, latitudes, color_map, 
                'CMA Improvement (%)', 'd')
    
    # Special formatting for CMA plot
    axes[1, 1].set_yticks([0, 20, 40, 60, 80])
    
    # Save and display
    plt.tight_layout()
    output_path = output_dir + filename
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {output_path}")
    #plt.show()

def main():
    """Main execution function."""
    # Configuration
    data_results_dir = './fct_data/'
    output_dir = './fct_data/'
    
    # Create the plots
    create_performance_plots(data_results_dir, output_dir)

if __name__ == "__main__":
    main()