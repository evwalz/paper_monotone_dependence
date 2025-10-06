import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from scipy.stats import rankdata
from scipy.special import comb
from numba import jit, prange

epsilon = 1e-3

def unique_with_counts_v2(arr):
    """Optimized version for large arrays - use numpy's built-in for 100k+ data"""
    return np.unique(arr, return_counts=True)

def prob_y_v2(y):
    """Optimized probability computation for large arrays"""
    unique, counts = unique_with_counts_v2(y)
    probabilities = counts / len(y)
    return probabilities[np.searchsorted(unique, y)]


def comp_rho_cma_v2(y_rank, x_rank):
    """Optimized correlation computations for large datasets"""
    N = len(y_rank)
    mean_rank = (N + 1) * 0.5
    
    # For large arrays, use numpy's optimized functions
    y_centered = y_rank - mean_rank
    x_centered = x_rank - mean_rank
    
    # Use numpy's var which is highly optimized for large arrays
    var_y = np.var(y_rank, ddof=1)
    
    # Optimized computations using dot products
    dot_product = np.dot(x_centered, y_centered)
    rho_val = (12.0 / (N * N * (N - 1))) * dot_product
    
    cov_xy = dot_product / (N - 1)
    cma_val = (cov_xy / var_y + 1) * 0.5
    
    return rho_val, cma_val


@jit(nopython=True, parallel=True, cache=True)
def mean_sign_x_numba(x_ranks_sorted, y_ranks_sorted, x_unique, y_unique):
    """
    Numba-compiled core computation - should be 10-50x faster
    """
    N = len(x_ranks_sorted)
    R = len(x_unique)
    M = len(y_unique)
        
    mean_sign_x = np.zeros((R, M), dtype=np.float64)
    inv_N = 1.0 / N
        
    # Parallel computation across unique values
    for i in prange(R):
        for j in range(M):
            sign_sum = 0.0
            for k in range(N):
                if x_ranks_sorted[k] > x_unique[i]:
                    sign_x = 1.0
                elif x_ranks_sorted[k] < x_unique[i]:
                    sign_x = -1.0
                else:
                    sign_x = 0.0
                    
                if y_ranks_sorted[k] > y_unique[j]:
                    sign_y = 1.0
                elif y_ranks_sorted[k] < y_unique[j]:
                    sign_y = -1.0
                else:
                    sign_y = 0.0
                    
                sign_sum += sign_x * sign_y
                
            mean_sign_x[i, j] = sign_sum * inv_N
        
    return mean_sign_x
    
def bivariate_mdf_expected_signbased_numba(x_ranks, y_ranks):
    """
     Numba-accelerated version for maximum performance
    """
    N = len(x_ranks)
        
    x_unique = np.unique(x_ranks)
    y_unique, y_counts = np.unique(y_ranks, return_counts=True)
        
    R = len(x_unique)
    M = len(y_unique)
        
    inv_N = 1.0 / N
    G_x_unique = (x_unique - 0.5) * inv_N
    G_y_unique = (y_unique - 0.5) * inv_N
        
    sort_idx = np.argsort(y_ranks)
    x_ranks_sorted = x_ranks[sort_idx]
    y_ranks_sorted = y_ranks[sort_idx]
        
    x_indices = np.searchsorted(x_unique, x_ranks_sorted)
    G_x_full = G_x_unique[x_indices]
        
    # Use numba for the core computation
    mean_sign_x = mean_sign_x_numba(x_ranks_sorted, y_ranks_sorted, x_unique, y_unique)
        
    # Rest of computation (vectorized)
    mean_sign_selected = mean_sign_x[x_indices, :]
    G_x_broadcast = G_x_full[:, np.newaxis]
    G_y_broadcast = G_y_unique[np.newaxis, :]
        
    all_exp = mean_sign_selected + 2 * G_x_broadcast + 2 * G_y_broadcast - 1
        
    y_positions = np.concatenate(([0], np.cumsum(y_counts)))
    g_1 = np.zeros(N)
        
    column_sums = np.sum(all_exp, axis=0) * inv_N
    for i in range(M):
        g_1[y_positions[i]:y_positions[i+1]] = column_sums[i]
        
    g_2 = np.sum(all_exp * y_counts[np.newaxis, :], axis=1) * inv_N
        
    g_1 *= 0.25
    g_2 *= 0.25
        
    reverse_idx = np.empty_like(sort_idx)
    reverse_idx[sort_idx] = np.arange(N)
        
    return g_1[reverse_idx], g_2[reverse_idx]

def del_our_second_single_numba(y_ranks, x_ranks): 
    N = len(y_ranks)
    inv_N = 1.0 / N

    zeta_3Y = 1-(12/N**2)*np.var(y_ranks)
    denum_zeta = 1 - zeta_3Y
    k_zeta = prob_y_v2(y_ranks) ** 2 - zeta_3Y

    rhos, cmas = comp_rho_cma_v2(y_ranks, x_ranks)
    g_1, g_2 = bivariate_mdf_expected_signbased_numba(x_ranks, y_ranks)
        
    rho_K_1 = (rhos * k_zeta) / denum_zeta
        
    Fbar = (x_ranks - 0.5) * inv_N
    Gbar = (y_ranks - 0.5) * inv_N
        
    K_1 = 4 * (g_1 + g_2 + Fbar * Gbar - Fbar - Gbar) + 1 - rhos
        
    temp = K_1 + rho_K_1
    sigma = (9 / (denum_zeta * denum_zeta)) * np.mean(temp * temp)
        
    return sigma / (4 * N)

def cma_test_sd(y, x):
    y_ranks = rankdata(y, method='average')
    x_ranks = rankdata(x, method='average')
    S = del_our_second_single_numba(y_ranks, x_ranks)
    return S

def cma(response, predictor):
    """
    Calculate CMA coefficient.
    """
    response = np.asarray(response)
    if response.ndim > 1:
        raise ValueError("CPA only handles 1-D arrays of responses")

    predictor = np.asarray(predictor)
	
    if predictor.ndim > 1:
        ValueError("CPA only handles 1-D arrays of forecasts")   
  
    	# check for nans
    if np.isnan(np.sum(response)) == True:
        ValueError("response contains nan values")
		
    if np.isnan(np.sum(predictor)) == True:
        ValueError("forecast contains nan values")
	                
    forecastRank = rankdata(predictor, method='average')
    responseRank = rankdata(response, method='average')
    
    return((np.cov(responseRank,forecastRank)[0][1]/np.cov(responseRank,responseRank)[0][1]+1)/2) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank_calibration_path', type=str, required=True,
                        help='Base path for rank calibration data')
    parser.add_argument('--root_dir', type=str, 
                       default=None,
                       help='Root directory containing calibration results')
    parser.add_argument('--output_dir', type=str, 
                       default=None,
                       help='Output directory for saving results')
    parser.add_argument('--correctness', type=str, default='rouge',
                       help='Correctness metric to use')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                       help='Model to evaluate')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Temperature parameter')
    parser.add_argument('--dataset', type=str, default='triviaqa',
                       help='Dataset to use')
    parser.add_argument('--mode', type=str, default='rougeL',
                       help='Evaluation mode')
    parser.add_argument('--metric', type=str, default='cma',
                       choices=['cma', 'erce'],
                       help='Metric to use for visualization (CMA or ERCE)')
    args = parser.parse_args()

    if args.root_dir is None:
        args.root_dir = os.path.join(args.rank_calibration_path, 'submission/calibration_results')
    if args.output_dir is None:
        args.output_dir = os.path.join(args.rank_calibration_path, 'stats_test')

    # Create output directory if it doesn't exist
    sys.path.insert(0, os.path.abspath(args.rank_calibration_path))
    from metrics import calibration
    os.makedirs(args.output_dir, exist_ok=True)

    # list all csv files in the root directory
    print(f"Loading files from {args.root_dir}")
    file_names = [file for file in os.listdir(args.root_dir) if file.endswith('.json')]
    model = args.model.split('/')[-1]
    
    # compute the correctness score
    scores_file = os.path.join(args.root_dir, f"{model}_{args.dataset}_{args.temperature}_{args.correctness}.json")
    if os.path.exists(scores_file):
        scores = json.load(open(scores_file))
    else:
        raise ValueError(f"File not found: {scores_file}")
    scores = pd.DataFrame(scores).dropna(axis=0)

    model = args.model.split('/')[-1]
    dataset = args.dataset
    file_names = []
    for method in ['whitebox', 'blackbox', 'verbalized']:
        if method == 'whitebox':
            affinity_mode = 'none'
            file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'whitebox']) + '.json'
            file_names.append(file_name)
        elif method == 'verbalized':
            try:
                file_name = "_".join(['calibrate', model, dataset, str(args.temperature), 'disagreement', 'verbalized']) + '.json'
                file_names.append(file_name)
            except:
                continue
        else:
            for affinity_mode in ['disagreement', 'agreement']:
                file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'blackbox']) + '.json'
                file_names.append(file_name)
    data_whitebox = json.load(open(os.path.join(args.root_dir, file_names[0])))
    data_blackbox_disagreement = json.load(open(os.path.join(args.root_dir, file_names[1])))
    data_blackbox_agreement = json.load(open(os.path.join(args.root_dir, file_names[2])))

    tmps = []
    #seeds = list(range(20))
    #for seed in seeds:
        #print(seed)
        #np.random.seed(seed)
    if model == 'gpt-3.5-turbo' and args.temperature == 1.0:
        data_verbalized = json.load(open(os.path.join(args.root_dir, file_names[3])))
        indices = np.arange(len(data_verbalized))#np.random.choice(len(data_verbalized), len(data_verbalized), replace=True).tolist()
        data_verbalized_bootstrap = [data_verbalized[index] for index in indices]
        
        for row_verbalized in tqdm(data_verbalized_bootstrap):
            tmp = {'model':model, 'dataset':dataset, 'metric':args.correctness, 'temperature':args.temperature}
            idx = row_verbalized['idx']
            row_whitebox = data_whitebox[idx]
            row_blackbox_disagreement = data_blackbox_disagreement[idx]
            row_blackbox_agreement = data_blackbox_agreement[idx]
                
            tmp['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
            tmp['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
            tmp['ecc_u_disagreement'] = [row_blackbox_disagreement['ecc_u']] * 10
            tmp['degree_u_disagreement'] = [row_blackbox_disagreement['degree_u']] * 10
            tmp['spectral_u_disagreement'] = [row_blackbox_disagreement['spectral_u']] * 10
            tmp['verbalized'] = row_verbalized['verbalized']

            tmp['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
            tmp['degree_c_agreement'] = row_blackbox_agreement['degree_c']
            tmp['ecc_u_agreement'] = [row_blackbox_agreement['ecc_u']] * 10
            tmp['degree_u_agreement'] = [row_blackbox_agreement['degree_u']] * 10
            tmp['spectral_u_agreement'] = [row_blackbox_agreement['spectral_u']] * 10

            tmp['entropy_normalized'] = [row_whitebox['entropy_normalized']] * 10
            tmp['entropy_unnormalized'] = [row_whitebox['entropy_unnormalized']] * 10
            tmp['normalized_nll_all'] = row_whitebox['normalized_nll']
            tmp['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']

                # select scores with the same index
            score = scores[scores['id'] == idx]
            tmp['normalized_score_all'] = score.iloc[0]['normalized_score']
            tmp['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
            normalized_min_index = np.argmin(tmp['normalized_nll_all'])
            unnormalized_min_index = np.argmin(tmp['unnormalized_nll_all'])
            tmp['normalized_score_greedy'] = tmp['normalized_score_all'][normalized_min_index]
            tmp['unnormalized_score_greedy'] = tmp['unnormalized_score_all'][unnormalized_min_index]
            tmps.append(tmp)
    else:
            # sample with replacement from the indices of the data
        #indices = np.random.choice(len(data_whitebox), len(data_whitebox), replace=True).tolist()
        indices = np.arange(len(data_whitebox)).tolist()
        data_whitebox_bootstrap = [data_whitebox[index] for index in indices]
        data_blackbox_disagreement_bootstrap = [data_blackbox_disagreement[index] for index in indices]
        data_blackbox_agreement_bootstrap = [data_blackbox_agreement[index] for index in indices]
        #tmps = []
        for idx, (index, row_whitebox, row_blackbox_disagreement, row_blackbox_agreement) in tqdm(enumerate(zip(indices, data_whitebox_bootstrap, data_blackbox_disagreement_bootstrap, data_blackbox_agreement_bootstrap)), total=len(data_whitebox)):
            tmp = {'model':model, 'dataset':dataset, 'metric':args.correctness, 'temperature':args.temperature}

            tmp['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
            tmp['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
            tmp['ecc_u_disagreement'] = [row_blackbox_disagreement['ecc_u']] * 10
            tmp['degree_u_disagreement'] = [row_blackbox_disagreement['degree_u']] * 10
            tmp['spectral_u_disagreement'] = [row_blackbox_disagreement['spectral_u']] * 10

            tmp['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
            tmp['degree_c_agreement'] = row_blackbox_agreement['degree_c']
            tmp['ecc_u_agreement'] = [row_blackbox_agreement['ecc_u']] * 10
            tmp['degree_u_agreement'] = [row_blackbox_agreement['degree_u']] * 10
            tmp['spectral_u_agreement'] = [row_blackbox_agreement['spectral_u']] * 10

            tmp['entropy_normalized'] = [row_whitebox['entropy_normalized']] * 10
            tmp['entropy_unnormalized'] = [row_whitebox['entropy_unnormalized']] * 10
            tmp['normalized_nll_all'] = row_whitebox['normalized_nll']
            tmp['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']

                # select scores with the same index
         
            score = scores[scores['id'] == index]
            tmp['normalized_score_all'] = score.iloc[0]['normalized_score']
            tmp['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
            normalized_min_index = np.argmin(tmp['normalized_nll_all'])
            unnormalized_min_index = np.argmin(tmp['unnormalized_nll_all'])
            tmp['normalized_score_greedy'] = tmp['normalized_score_all'][normalized_min_index]
            tmp['unnormalized_score_greedy'] = tmp['unnormalized_score_all'][unnormalized_min_index]
            tmps.append(tmp)

    df = pd.DataFrame(tmps).dropna(axis=0)

    if model == 'gpt-3.5-turbo' and args.temperature == 1.0:
        uncertainty_indicators = ['ecc_u_agreement', 'degree_u_agreement', 'spectral_u_agreement', 'verbalized',
                                    'normalized_nll_all', 'unnormalized_nll_all', 'entropy_normalized', 'entropy_unnormalized']
    else:
        uncertainty_indicators = ['ecc_u_agreement', 'degree_u_agreement', 'spectral_u_agreement', 
                                    'normalized_nll_all', 'unnormalized_nll_all', 'entropy_normalized', 'entropy_unnormalized']
        
    correctness_scores = np.stack(df['normalized_score_all']).flatten()
    result = {'model': model, 'dataset': dataset, 'metric': args.correctness, 'temperature': args.temperature}

    # run numba function to make it available:
    x = np.random.normal(loc=0, scale=1, size=50)
    y = np.random.normal(loc=0, scale=1, size=50)
    sd_value = cma_test_sd(y, x)
        
    for indicator in uncertainty_indicators:
        print(indicator)
        uncertainty = np.stack(df[indicator]).flatten() if 'verbalized' not in indicator else -np.stack(df[indicator]).flatten()
        #erce = calibration.plugin_RCE_est(correctness=correctness_scores, uncertainties=uncertainty, num_bins=20, p=1)
        #result[f'{indicator}_erce'] = erce

        #print(len(correctness_scores))
        #print(len(np.unique(correctness_scores)))
       # print(len(np.unique(uncertainty)))
            # compute CMA
        #cma_val = cma(response=correctness_scores, predictor=-uncertainty)
        #result[f'{indicator}_cma'] = cma_val

        # compute p-values
        sd_value = cma_test_sd(correctness_scores, -uncertainty)
        result[f'{indicator}_sd'] = sd_value
    
        #results.append(tmp)
    
    # Update the output file path to use the specified output directory
    output_file = os.path.join(args.output_dir, f'{model}_{args.dataset}_{args.temperature}_{args.correctness}_stat_test.json')
    with open(output_file, 'w') as f:
        json.dump(result, f)
    