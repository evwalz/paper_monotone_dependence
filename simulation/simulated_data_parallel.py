import numpy as np
from scipy.stats import norm
from helpers import cma_stat_test, cma_stat_test_spear
from tqdm import tqdm
import argparse
import sys
import multiprocessing as mp
from functools import partial
import os

# Define default sample sizes for each experiment type
DEFAULT_BINARY_SAMPLE_SIZES = [50, 100, 500, 1000, 5000]
DEFAULT_REAL_SAMPLE_SIZES = [10, 50, 100, 500, 1000]
DEFAULT_DISCRETIZED_SAMPLE_SIZES = [10, 50, 100, 500, 1000]

def get_sample_sizes(experiment_type):
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
        return DEFAULT_BINARY_SAMPLE_SIZES

# Helper function for parallel processing in binary simulation
def binary_sim_worker(args):
    i, n, sigma_1, sigma_2 = args
    w_1 = np.random.normal(0, sigma_1, n)
    w_2 = np.random.normal(0, sigma_2, n)
    # Vectorize calculations
    p_1 = norm.cdf(w_1 / np.sqrt(1+sigma_2**2))
    p_2 = norm.cdf(w_2 / np.sqrt(1+sigma_1**2))
    p_3 = norm.cdf(w_1 + w_2)
    y0 = np.random.binomial(1, p_3, n)
    x0 = np.vstack((p_1, p_2))
    obj = cma_stat_test(y0, x0)
    return obj.global_p

# Helper function for parallel processing in real simulation
def real_sim_worker(args):
    i, n, sigma_1, sigma_2 = args
    X_0 = np.random.normal(0, 1, n)
    Z_1 = np.random.normal(0, 1, n)
    Z_2 = np.random.normal(0, 1, n)
    Y_0 = np.random.normal(X_0, 1, n)
    # Vectorize calculations
    X_1 = X_0 + Z_1
    X_2 = X_0 + Z_2
    X = np.vstack((X_1, X_2))
    obj = cma_stat_test(Y_0, X)
    return obj.global_p

# Helper function for parallel processing in discretized simulation
def discretized_sim_worker(args):
    i, n, sigma_1, sigma_2 = args
    X_0 = np.random.normal(0, 1, n)
    Z_1 = np.random.normal(0, 1, n)
    Z_2 = np.random.normal(0, 1, n)
    Y_0 = np.round(np.random.normal(X_0, 1, n))
    # Vectorize calculations
    X_1 = np.round(X_0 + Z_1)
    X_2 = np.round(X_0 + Z_2)
    X = np.vstack((X_1, X_2))
    obj = cma_stat_test(Y_0, X)
    return obj.global_p
# Helper function for parallel processing in discretized simulation
def discretized_spear_sim_worker(args):
    i, n, sigma_1, sigma_2 = args
    X_0 = np.random.normal(0, 1, n)
    Z_1 = np.random.normal(0, 1, n)
    Z_2 = np.random.normal(0, 1, n)
    Y_0 = np.round(np.random.normal(X_0, 1, n))
    # Vectorize calculations
    X_1 = np.round(X_0 + Z_1)
    X_2 = np.round(X_0 + Z_2)
    X = np.vstack((X_1, X_2))
    obj = cma_stat_test_spear(Y_0, X)
    return obj.global_p

def simulate_pvals_binary(n, T=1000, output_dir='./simulation_results/'):
    sigma_1 = 3
    sigma_2 = sigma_1
    
    # Set up multiprocessing
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    # Create parameter list for each iteration
    args_list = [(i, n, sigma_1, sigma_2) for i in range(T)]
    
    # Run simulations in parallel
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to show progress
        ps = list(tqdm(pool.imap(binary_sim_worker, args_list), total=T))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, f'binary_p_n{n}_rep{T}.txt')
    np.savetxt(output_file, np.asarray(ps))

def simulate_pvals_real(n, T=1000, output_dir='./simulation_results/'):
    sigma_1 = 3
    sigma_2 = sigma_1
    
    # Set up multiprocessing
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    # Create parameter list for each iteration
    args_list = [(i, n, sigma_1, sigma_2) for i in range(T)]
    
    # Run simulations in parallel
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to show progress
        ps = list(tqdm(pool.imap(real_sim_worker, args_list), total=T))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, f'real_p_n{n}_rep{T}.txt')
    np.savetxt(output_file, np.asarray(ps))

def simulate_pvals_discretized(n, T=1000, output_dir='./simulation_results/'):
    sigma_1 = 3
    sigma_2 = sigma_1
    
    # Set up multiprocessing
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    print(f"Using {num_cores} cores for parallel processing")
    
    # Create parameter list for each iteration
    args_list = [(i, n, sigma_1, sigma_2) for i in range(T)]
    
    # Run simulations in parallel
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to show progress
        ps = list(tqdm(pool.imap(discretized_sim_worker, args_list), total=T))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, f'discretized_p_n{n}_rep{T}.txt')
    np.savetxt(output_file, np.asarray(ps))


def simulate_pvals_discretized_spear(n, T=1000, output_dir='./simulation_results/'):
    sigma_1 = 3
    sigma_2 = sigma_1
    
    # Set up multiprocessing
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    print(f"Using {num_cores} cores for parallel processing")
    
    # Create parameter list for each iteration
    args_list = [(i, n, sigma_1, sigma_2) for i in range(T)]
    
    # Run simulations in parallel
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to show progress
        ps = list(tqdm(pool.imap(discretized_spear_sim_worker, args_list), total=T))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, f'discretized_spear_p_n{n}_rep{T}.txt')
    np.savetxt(output_file, np.asarray(ps))

def main():
    parser = argparse.ArgumentParser(description='Optimized statistical simulation code')
    parser.add_argument('--output_dir', type=str, default='./simulated_data/', help='Output file path')
    #parser.add_argument('--n', type=int, default=100, help='sample sizes per repetition')
    parser.add_argument('--T', type=int, default=100, help='number of repetition of simulated example')
    parser.add_argument('--experiment', type=str, default='binary', choices=['binary', 'real', 'discretized', 'discretized_spear'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    sample_sizes = get_sample_sizes(args.experiment)
    
    for n in sample_sizes:
        if args.experiment == 'binary':
            simulate_pvals_binary(n, args.T, args.output_dir)
        elif args.experiment == 'real':
            simulate_pvals_real(n, args.T, args.output_dir)
        elif args.experiment == 'discretized':
            simulate_pvals_discretized(n, args.T, args.output_dir)
        elif args.experiment == 'discretized_spear':
            simulate_pvals_discretized_spear(n, args.T, args.output_dir)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())