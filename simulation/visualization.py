import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
from scipy import stats
import pandas as pd
import seaborn as sns

# Define default sample sizes for each experiment type
DEFAULT_BINARY_SAMPLE_SIZES = [50, 100, 500, 1000, 5000]
DEFAULT_REAL_SAMPLE_SIZES = [10, 50, 100, 500, 1000]
DEFAULT_DISCRETIZED_SAMPLE_SIZES = [10, 50, 100, 500, 1000]

def get_default_sample_sizes(experiment_type):
    """
    Get default sample sizes for different experiment types.
    
    Parameters:
    -----------
    experiment_type : str
        Type of experiment ('binary', 'real', or 'discretized')
        
    Returns:
    --------
    list:
        Default sample sizes for the specified experiment type
    """
    if experiment_type == 'binary':
        return DEFAULT_BINARY_SAMPLE_SIZES
    elif experiment_type == 'real':
        return DEFAULT_REAL_SAMPLE_SIZES
    elif experiment_type == 'discretized':
        return DEFAULT_DISCRETIZED_SAMPLE_SIZES
    elif experiment_type == 'discretized_spear':
        return DEFAULT_DISCRETIZED_SAMPLE_SIZES
    else:
        # Default for unknown experiment types
        return DEFAULT_BINARY_SAMPLE_SIZES

def plot_pvalue_distributions(experiment_type, sample_sizes, T, input_dir, output_dir, 
                             bins=20, figsize=None, show_uniform=True):
    """
    Plot histograms of p-values for different sample sizes.
    
    Parameters:
    -----------
    experiment_type : str
        Type of experiment ('binary', 'real', or 'discretized')
    sample_sizes : list
        List of sample sizes to plot
    T : int
        Number of repetitions used in simulations
    input_dir : str
        Directory where simulation results are stored
    output_dir : str
        Directory to save the plot
    bins : int, optional
        Number of bins for histograms
    figsize : tuple, optional
        Figure size (width, height)
    show_uniform : bool, optional
        If True, add a horizontal line at y=1 to show the uniform distribution
    """
    if figsize is None:
        figsize = (5*len(sample_sizes), 5)
        
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=figsize)
    
    # Handle case where only one sample size is provided
    if len(sample_sizes) == 1:
        axes = [axes]

    
    
    for i, n in enumerate(sample_sizes):
        try:
            # Construct filename based on experiment type
            if experiment_type == 'binary':
                filename = f'{input_dir}/binary_p_n{n}_rep{T}.txt'
            elif experiment_type == 'real':
                filename = f'{input_dir}/real_p_n{n}_rep{T}.txt'
            elif experiment_type == 'discretized':
                filename = f'{input_dir}/discretized_p_n{n}_rep{T}.txt'
            elif experiment_type == 'discretized_spear':
                filename = f'{input_dir}/discretized_spear_p_n{n}_rep{T}.txt'
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
            
            # Load p-values
            ps = np.loadtxt(filename)
            
            # Create histogram
            axes[i].hist(np.asarray(ps), bins=bins, color="darkgrey", 
                      edgecolor='black', density=True, linewidth=1.2)
            
            if show_uniform:
                axes[i].axhline(1, color='black', linestyle='--', label='Uniform')
                
            axes[i].set_title(f'n = {n}')

            if i == 0:
                counts, _ = np.histogram(ps, bins=bins, density=True)
                y_max = max(counts) * 1.05

            if experiment_type == 'discretized' or experiment_type == 'discretized_spear':
                y_max = 2.05

            axes[i].set_ylim(0, y_max)
            #axes[i].set_xlabel('p-value')
            
            # Only add y-label to the first subplot
            #if i == 0:
            #    axes[i].set_ylabel('Density')
                
        except FileNotFoundError:
            print(f"Warning: File not found for n={n}, experiment={experiment_type}")
            axes[i].text(0.5, 0.5, f"No data for n={n}", 
                       horizontalalignment='center', verticalalignment='center')
            axes[i].set_title(f'n = {n}')
    
    # Add a main title
    #plt.suptitle(f'Distribution of p-values for {experiment_type} experiment (T={T})', 
    #            fontsize=16, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, f'{experiment_type}_pvalue_distributions_T{T}.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot simulation results')
    parser.add_argument('--input_dir', type=str, default='./simulated_data/', 
                       help='Directory where simulation results are stored')
    parser.add_argument('--output_dir', type=str, default='./plots/', 
                       help='Directory to save plots')
    parser.add_argument('--T', type=int, default=100, 
                       help='Number of repetitions used in simulations')
    parser.add_argument('--experiment', type=str, default='binary',
                       choices=['binary', 'real', 'discretized', 'discretized_spear'],
                       help='Type of experiment to plot')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which experiment types to process
    experiment_types = [args.experiment]
    
    # Process each experiment type
    for exp_type in experiment_types:
        # Get default sample sizes for this experiment type
        sample_sizes = get_default_sample_sizes(exp_type)
        
        # Generate requested plots
        plot_pvalue_distributions(exp_type, sample_sizes, args.T, args.input_dir, args.output_dir)
    
    print("Plotting completed successfully.")

if __name__ == "__main__":
    main()