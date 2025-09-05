import numpy as np
import matplotlib.pyplot as plt
from helpers import run_simulation_meng_our


def compute_pvalue_data(n_values, T, discrete=True, alternative='two.sided'):
    """
    Compute p-values for different sample sizes.
    
    Parameters:
    -----------
    n_values : list
        List of sample sizes to test
    T : int
        Number of simulations to run
    discrete : bool
        Whether to use discrete simulation
    alternative : str
        Type of alternative hypothesis ('two.sided' or 'one.sided')
    
    Returns:
    --------
    tuple
        (list_of_pvals_meng, list_of_pvals_our) - arrays of p-values
    """
    num_n_values = len(n_values)
    list_of_pvals_meng = np.zeros((num_n_values, T))
    list_of_pvals_our = np.zeros((num_n_values, T))
    
    # Run simulations for each sample size
    for i, n in enumerate(n_values):
        print(f"Processing n = {n}...")
        
        # Run simulation
        pvals_meng, pvals_our = run_simulation_meng_our(
            n, T=T, discrete=discrete, alternative=alternative
        )
        
        # Store results
        list_of_pvals_meng[i, :] = pvals_meng
        list_of_pvals_our[i, :] = pvals_our
    
    return list_of_pvals_meng, list_of_pvals_our

def create_pvalue_histograms(n_values, list_of_pvals_meng, list_of_pvals_our, 
                           discrete=True, alternative='two.sided', filename_suffix=None):
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
    filename_suffix : str, optional
        Suffix to add to filename
    """
    num_n_values = len(n_values)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, num_n_values, figsize=(20, 8))
    
    # Ensure axes is 2D array even for single column
    if num_n_values == 1:
        axes = axes.reshape(2, 1)
    
    # Create histograms for each sample size
    for i, n in enumerate(n_values):
        # Create histograms
        axes[0, i].hist(
            list_of_pvals_meng[i, :], bins=20, density=True, alpha=1, color="darkgrey",
            edgecolor='black', linewidth=1.2, label='Meng Method'
        )
        axes[1, i].hist(
            list_of_pvals_our[i, :], bins=20, density=True, alpha=1, color="darkgrey",
            edgecolor='black', linewidth=1.2, label='Our Method'
        )
        
        # Format subplots
        for j in range(2):
            axes[j, i].axhline(y=1, color='black', linestyle='--', linewidth=1)
            axes[j, i].set_title(f'n = {n}')
            axes[j, i].set_ylim(0, 1.2)
            axes[j, i].set_xlabel('P-value')
            axes[j, i].set_ylabel('Density')
            axes[j, i].grid(True, alpha=0.3)
    
    # Add method labels
    axes[0, 0].text(-0.03, 1.05, 'Meng', transform=axes[0, 0].transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='left')
    axes[1, 0].text(-0.03, 1.05, 'Our', transform=axes[1,0].transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='left')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for main title
    
    # Save as PDF
    discrete_str = "discrete" if discrete else "real"
    filename = f"pvalue_histograms_{discrete_str.lower()}_{alternative.replace('.', '_')}"
    if filename_suffix:
        filename += f"_{filename_suffix}"
    filename += ".pdf"
    plt.savefig('./' + filename, format='pdf', dpi=300, bbox_inches='tight')
    
    return None


def run_pvalue_analysis(n_values, T, discrete=True, alternative='two.sided', filename_suffix=None):

    list_of_pvals_meng, list_of_pvals_our = compute_pvalue_data(
        n_values, T, discrete, alternative
    )

    create_pvalue_histograms(
        n_values, list_of_pvals_meng, list_of_pvals_our, 
        discrete, alternative, filename_suffix
    )
    
    return list_of_pvals_meng, list_of_pvals_our


def main():
    """Main function to run the analysis."""
    
    # Configuration
    n_values = [50, 100, 500, 1000, 5000]
    T = 100000
    
    # Run discrete two-sided test
    results_discrete_two = run_pvalue_analysis(
        n_values=n_values,
        T=T,
        discrete=True,
        alternative='two.sided'
        )

    
    # Run continuous one-sided test
    results_continuous_one = run_pvalue_analysis(
        n_values=n_values,
        T=T,
        discrete=False,
        alternative='one.sided'
    )
    
    return results_discrete_two, results_continuous_one


if __name__ == "__main__":
    # Run the analysis
    discrete_results, continuous_results = main()
