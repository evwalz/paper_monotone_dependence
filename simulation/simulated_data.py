import numpy as np
from scipy.stats import norm
from simulation.helpers import cma_stat_test
from tqdm import tqdm
from argparse import argparse
import sys

def simulate_pvals_binary(n, T = 1000, output_dir = './simulation_results/'):
    sigma_1 = 3
    sigma_2 = sigma_1
    ps = []
    for i in tqdm(range(T)):
        w_1 = np.random.normal(0, sigma_1, n)
        w_2 = np.random.normal(0, sigma_2, n)
        p_1 = norm.cdf(w_1 / (np.sqrt(1+sigma_2**2)))
        p_2 = norm.cdf(w_2 / (np.sqrt(1+sigma_1**2)))
        p_3 = norm.cdf(w_1 + w_2)
        y0 = np.random.binomial(1, p_3, n)
        x0 = np.vstack((p_1, p_2)) 
        obj = cma_stat_test(y0, x0)
        ps.append(obj.global_p)
    np.savetxt(output_dir + 'binary_p_n'+str(n)+'_rep_'+str(T)+'.txt', np.asarray(ps))

def simulate_pvals_real(n, T = 1000, output_dir = './simulation_results/'):
    sigma_1 = 3
    sigma_2 = sigma_1
    ps = []
    for i in tqdm(range(T)):
        X_0 = np.random.normal(0, 1, n)
        Z_1 = np.random.normal(0, 1, n)
        Z_2 = np.random.normal(0, 1, n)
        Y_0 = np.random.normal(X_0, 1, n)
        X_1 = X_0 + Z_1
        X_2 = X_0 + Z_2   
        X = np.vstack((X_1, X_2))
        obj = cma_stat_test(Y_0, X)
        ps.append(obj.global_p)
    np.savetxt(output_dir + 'real_p_n'+str(n)+'_rep_'+str(T)+'.txt', np.asarray(ps))


def simulate_pvals_discretized(n, T = 1000, output_dir = './simulation_results/'):
    sigma_1 = 3
    sigma_2 = sigma_1
    ps = []
    for i in tqdm(range(T)):
        X_0 = np.random.normal(0, 1, n)
        Z_1 = np.random.normal(0, 1, n)
        Z_2 = np.random.normal(0, 1, n)
        Y_0 = np.round(np.random.normal(X_0, 1, n))
        X_1 = np.round(X_0 + Z_1)
        X_2 = np.round(X_0 + Z_2)   
        X = np.vstack((X_1, X_2))
        obj = cma_stat_test(Y_0, X)
        ps.append(obj.global_p)
    np.savetxt(output_dir + 'discretized_p_n'+str(n)+'_rep_'+str(T)+'.txt', np.asarray(ps))


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--output_dir', type=str, default='./simulated_data/', help='Output file path')
    parser.add_argument('--n', type=int, default=100, help='sample sizes per repetition')
    parser.add_argument('--T', type=int, default=100, help='number of repetition of simulated example')
    parser.add_argument('--experiment', type=str, default='binary',choices = ['binary', 'real', 'discretized'])
    args = parser.parse_args()
    if args.experiment == 'binary':
        simulate_pvals_binary(args.n, args.T, args.output_dir)
    elif args.experiment == 'real':
        simulate_pvals_real(args.n, args.T, args.output_dir)
    elif args.experiment == 'discretized':
        simulate_pvals_discretized(args.n, args.T, args.output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())

