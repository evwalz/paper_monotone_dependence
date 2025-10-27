import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def load_results(n_values, discrete, alternative, results_dir='./results'):
    """
    Load saved p-value results from files.
    
    Parameters:
    -----------
    n_values : list
        List of sample sizes
    discrete : bool
        Whether to load discrete results
    alternative : str
        Type of alternative hypothesis
    results_dir : str
        Directory containing result files
    
    Returns:
    --------
    tuple
        (list_of_pvals_meng, list_of_pvals_our)
    """
    discrete_str = "discrete" if discrete else "continuous"
    alt_str = alternative.replace('.', '_')
    
    num_n_values = len(n_values)
    list_of_pvals_meng = []
    list_of_pvals_our = []
    
    for n in n_values:
        filename_meng = os.path.join(results_dir, f"pvals_meng_{discrete_str}_{alt_str}_n{n}.npy")
        filename_our = os.path.join(results_dir, f"pvals_our_{discrete_str}_{alt_str}_n{n}.npy")
        
        # Load our method results (always present)
        if os.path.exists(filename_our):
            pvals_our = np.load(filename_our)
            list_of_pvals_our.append(pvals_our)
            print(f"Loaded our method: n={n}, shape={pvals_our.shape}")
        else:
            raise FileNotFoundError(f"Missing file: {filename_our}")
        
        # Load Meng method results (only for continuous)
        if not discrete:
            if os.path.exists(filename_meng):
                pvals_meng = np.load(filename_meng)
                list_of_pvals_meng.append(pvals_meng)
                print(f"Loaded Meng method: n={n}, shape={pvals_meng.shape}")
            else:
                raise FileNotFoundError(f"Missing file: {filename_meng}")
        else:
            list_of_pvals_meng.append(np.array([]))
    
    # Convert to numpy arrays
    if discrete:
        list_of_pvals_meng = np.array([np.array([]) for _ in range(num_n_values)])
    else:
        list_of_pvals_meng = np.array(list_of_pvals_meng)
    
    list_of_pvals_our = np.array(list_of_pvals_our)
    
    return list_of_pvals_meng, list_of_pvals_our


def create_pvalue_histograms(n_values, list_of_pvals_meng, list_of_pvals_our, 
                           discrete=True, alternative='two.sided', output_dir='./plots'):
    """
    Create histograms from pre-computed p-values.
    
    Parameters:
    -----------
    n_values : list
        List of sample sizes
    list_of_pvals_meng : numpy array
        P-values from Meng method (shape: num_n_values x T)
    list_of_pvals_our : numpy array
        P-values from our method (shape: num_n_values x T)
    discrete : bool
        Whether data is from discrete simulation
    alternative : str
        Type of alternative hypothesis
    output_dir : str
        Directory to save plots
    """
    num_n_values = len(n_values)
    
    # Create figure with subplots
    num_rows = 1 if discrete else 2
    
    fig, axes = plt.subplots(num_rows, num_n_values, figsize=(20, 4 * num_rows))
    fs = 20
    plt.rcParams.update({'font.size': 16})
    # Ensure axes is 2D array even for single row or column
    if num_rows == 1 and num_n_values == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_n_values == 1:
        axes = axes.reshape(-1, 1)
    
    # Create histograms for each sample size
    for i, n in enumerate(n_values):
        if not discrete:
            # Create Meng histogram
            axes[0, i].hist(
                list_of_pvals_meng[i, :], bins=20, density=True, alpha=1, color="darkgrey",
                edgecolor='black', linewidth=1.2, label='Meng Method'
            )
            axes[0, i].axhline(y=1, color='black', linestyle='--', linewidth=1)
            axes[0, i].set_title(f'n = {n}')
            axes[0, i].set_ylim(0, 1.2)
            axes[0, i].set_xticks([0, 0.5, 1])
            axes[0, i].set_yticks([0, 0.5, 1])
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_title(f'n = {n}', fontsize=fs)
            axes[0, i].tick_params(axis='both', which='major', labelsize=fs)
        
        # Create Our method histogram
        row_idx = 1 if not discrete else 0
        axes[row_idx, i].hist(
            list_of_pvals_our[i, :], bins=20, density=True, alpha=1, color="darkgrey",
            edgecolor='black', linewidth=1.2, label='Our Method'
        )
        axes[row_idx, i].axhline(y=1, color='black', linestyle='--', linewidth=1)
        axes[row_idx, i].set_title(f'n = {n}')
        axes[row_idx, i].set_ylim(0, 1.2)
        axes[row_idx, i].set_xticks([0, 0.5, 1])
        axes[row_idx, i].set_yticks([0, 0.5, 1])
        axes[row_idx, i].grid(True, alpha=0.3)
        axes[row_idx, i].set_title(f'n = {n}', fontsize=fs)
        axes[row_idx, i].tick_params(axis='both', which='major', labelsize=fs)
    
    # Add method labels
    if not discrete:
        axes[0, 0].text(-0.03, 1.05, 'Meng', transform=axes[0, 0].transAxes, 
                       fontsize=fs, verticalalignment='bottom', horizontalalignment='left')
        axes[1, 0].text(-0.03, 1.05, 'Our', transform=axes[1, 0].transAxes, 
                       fontsize=fs, verticalalignment='bottom', horizontalalignment='left')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save as PDF
    os.makedirs(output_dir, exist_ok=True)
    discrete_str = "discrete" if discrete else "continuous"
    filename = f"pvalue_histograms_{discrete_str.lower()}_{alternative.replace('.', '_')}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filepath}")
    
    plt.close()


def main():
    """Main function to create plots from saved results."""
    
    # Configuration
    n_values = [50, 100, 500, 1000, 5000]
    results_dir = './results'
    plots_dir = results_dir
    
    print("Creating plots from saved results...")
    print("=" * 60)
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # List available result files
    result_files = glob.glob(os.path.join(results_dir, "*.npy"))
    print(f"Found {len(result_files)} result files")
    
    # Create discrete two-sided plot
    print("\n--- Processing discrete two-sided results ---")
    try:
        list_of_pvals_meng, list_of_pvals_our = load_results(
            n_values, discrete=True, alternative='two.sided', results_dir=results_dir
        )
        create_pvalue_histograms(
            n_values, list_of_pvals_meng, list_of_pvals_our,
            discrete=True, alternative='two.sided', output_dir=plots_dir
        )
        print("✓ Discrete two-sided plot created")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
    
    # Create continuous one-sided plot
    print("\n--- Processing continuous one-sided results ---")
    try:
        list_of_pvals_meng, list_of_pvals_our = load_results(
            n_values, discrete=False, alternative='one.sided', results_dir=results_dir
        )
        create_pvalue_histograms(
            n_values, list_of_pvals_meng, list_of_pvals_our,
            discrete=False, alternative='one.sided', output_dir=plots_dir
        )
        print("✓ Continuous one-sided plot created")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Plotting complete!")
    print(f"Plots saved in: {plots_dir}/")


if __name__ == "__main__":
    main()
