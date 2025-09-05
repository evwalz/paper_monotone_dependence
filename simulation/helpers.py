import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import spearmanr
from scipy.stats import rankdata
from scipy.special import comb
from numba import njit, prange


from scipy import stats
from tqdm import tqdm


# Helper function to calculate combinations (n choose k)
@njit
def comb(n, k):
    if n < k:
        return 0
    if k == 0:
        return 1
    num = 1
    denom = 1
    for i in range(k):
        num *= n - i
        denom *= i + 1
    return num // denom  # Using integer division to avoid float results

# JIT compilation with numba for kernel_p
# kernel_p(xarray_ranks[j, :], y_rank, rhos[j],x_unique, x_num, y_unique, y_num)


@njit(parallel=True)
def kernel_ties_optim2(x_rank_sort, y_rank_sort, rho, y_rank_unique, y_num, pos_y, ix_y, x_rank_unique):
    #y_rank_sort = y_rank[ix_y]
    #x_rank_sort = x_rank[ix_y]

    N = len(x_rank_sort)
    R = len(x_rank_unique)  # The number of unique x_rank values (much smaller than N)
    M = len(y_rank_unique)

    # Precompute G_x for unique x_ranks and G_y for unique y_ranks
    G_x_unique = (x_rank_unique - 0.5) / N
    G_y = (y_rank_unique - 0.5) / N
    

    # Map the unique G_x values back to the sorted x_ranks
    G_x = np.zeros(N)
    x_rank_indices = np.searchsorted(x_rank_unique, x_rank_sort)
    for i in range(N):
        G_x[i] = G_x_unique[x_rank_indices[i]]

    # Precompute mean sign comparison for all unique x_rank combinations
    mean_sign_x = np.zeros((R, M))
    
    for i in prange(R):
        for j in range(M):
            sign_sum = 0.0
            for k in range(N):
                sign_sum += np.sign(x_rank_sort[k] - x_rank_unique[i]) * np.sign(y_rank_sort[k] - y_rank_unique[j])
            mean_sign_x[i, j] = sign_sum / N

    # Initialize all_exp using the precomputed mean_sign_x
    all_exp = np.zeros((N, M))

    # Assign precomputed mean_sign_x back to all_exp using unique x_rank indices
    for i in prange(N):
        for j in range(M):
            all_exp[i, j] = mean_sign_x[x_rank_indices[i], j] + 2 * G_x[i] + 2 * G_y[j] - 1

    # Initialize g_1 and g_2 to zeros
    g_1 = np.zeros(N)
    g_2 = np.zeros(N)

    # Compute g_1 for each group of tied y_ranks
    for i in prange(M):
        total_sum = 0.0
        for j in range(N):
            total_sum += all_exp[j, i]
        for k in range(pos_y[i], pos_y[i+1]):
            g_1[k] = total_sum / N

    # Compute g_2 by summing across axis=1, weighted by y_num
    for i in prange(N):
        total_sum = 0.0
        for j in range(M):
            total_sum += all_exp[i, j] * y_num[j]
        g_2[i] = total_sum / N

    # Divide g_1 and g_2 by 4, as per the original formula
    g_1 /= 4
    g_2 /= 4

    # Expand G_y from unique values to all y_rank_sort elements (corresponding to the sorted y_ranks)
    G_y_full = (y_rank_sort - 0.5) / N
    
    # Compute k_p using the precomputed values
    k_p = np.zeros(N)
    for i in prange(N):
        k_p[i] = 4 * (g_1[i] + g_2[i] + G_x[i] * G_y_full[i] - G_y_full[i] - G_x[i]) + 1 - rho

    # Reverse the sorting order to match the original input
    rev_order = np.argsort(ix_y)
    k_p = k_p[rev_order]  # Unsort to original order
    
    return k_p


# Manual implementation of unique and counts since numba doesn't support np.unique with return_counts
@njit
def unique_with_counts(arr):
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]
    
    unique_vals = []
    counts = []
    
    count = 1
    for i in range(1, len(arr)):
        if sorted_arr[i] == sorted_arr[i - 1]:
            count += 1
        else:
            unique_vals.append(sorted_arr[i - 1])
            counts.append(count)
            count = 1
    
    # Add the last element
    unique_vals.append(sorted_arr[-1])
    counts.append(count)
    
    return np.array(unique_vals), np.array(counts)

# Numba-optimized zeta function
@njit
def zeta_fun(y):
    N = len(y)
    if N < 3:
        return 0
    
    unique, counts = unique_with_counts(y)
    triplets_count = 0
    for count in counts:
        if count >= 3:
            triplets_count += comb(count, 3)

    bin_N_3 = 6 / (N * (N - 1) * (N - 2))
    return bin_N_3 * triplets_count

# Optimized probability computation using numba
@njit
def prob_y(y):
    unique, counts = unique_with_counts(y)
    probabilities = counts / len(y)
    return probabilities[np.searchsorted(unique, y)]

# Numba-optimized function to compute rho and cma values
@njit
def comp_rho_cma(y_rank, x_rank):
    N = len(y_rank)
    mean_rank = (N + 1) / 2
    var_y =  np.sum((y_rank - np.mean(y_rank))**2)*(1/(N-1))
    rho_val = (12 / (N ** 2)) * (1 / (N - 1)) * np.sum((x_rank - mean_rank) * (y_rank - mean_rank))
    cma_val = (np.cov(y_rank, x_rank)[0, 1] / var_y + 1) / 2
    return rho_val, cma_val

def Sigma_fast2(y_rank, xarray_ranks):
    N = len(y_rank)
    k = xarray_ranks.shape[0]

    zeta_3Y = zeta_fun(y_rank)
    k_zeta = prob_y(y_rank) ** 2 - zeta_3Y
    sigma_zeta = 9 * np.mean(k_zeta ** 2)

    rhos = np.zeros(k)
    cmas = np.zeros(k)
    kps = np.zeros((k, N))

    y_rank_unique, y_num = np.unique(y_rank, return_counts=True)
    pos_y = np.insert(np.cumsum(y_num), 0, 0)
    
    # Sort based on y_rank
    ix_y = np.argsort(y_rank)
    #y_rank_unique, y_rank_inverse = np.unique(y_rank, return_inverse=True)
    y_rank_sort = y_rank[ix_y]
    
    
    # Get unique x_rank values and their counts
    # Precompute rho and CMA for all ranks
    for j in range(k):
        rhos[j], cmas[j] = comp_rho_cma(y_rank, xarray_ranks[j, :])
        #x_rank_unique, x_rank_inverse = np.unique(xarray_ranks[j, :], return_inverse=True)

        x_rank_sort = xarray_ranks[j, ix_y]
        x_rank_unique = np.unique(x_rank_sort)
        kps[j,:] = kernel_ties_optim2(x_rank_sort, y_rank_sort, rhos[j], y_rank_unique, y_num, pos_y, ix_y, x_rank_unique)

    factor = 1 / ((1 - zeta_3Y) ** 2)
    phalf = np.zeros(k)
    S = np.zeros((k, k))

    for j in prange(k):
        k_p = kps[j, :]#kernel_p(xarray_ranks[j, :], y_rank, rhos[j])
        sigma_rho = 9 * np.mean(k_p ** 2)
        sigma_pz = 9 * np.mean(k_p * k_zeta)
        
        var = factor * (
            sigma_rho + (2 * rhos[j] * sigma_pz) / (1 - zeta_3Y) +
            (rhos[j] ** 2 * sigma_zeta) / ((1 - zeta_3Y) ** 2)
        )
        S[j, j] = var / (4 * N)
        phalf[j] = 1 - norm.cdf((cmas[j] - 0.5) / np.sqrt(var / (4 * N)))

        # Calculate off-diagonal elements
        for i in prange(j + 1, k):
            k_p2 = kps[i, :]#kernel_p(xarray_ranks[i, :], y_rank, rhos[i])
            sigma_rho2 = 9 * np.mean(k_p * k_p2)
            sigma_pz2 = 9 * np.mean(k_p2 * k_zeta)
            
            var = factor * (
                sigma_rho2 + (rhos[j] * sigma_pz) / (1 - zeta_3Y) +
                (rhos[i] * sigma_pz2) / (1 - zeta_3Y) +
                (rhos[j] * rhos[i] * sigma_zeta) / ((1 - zeta_3Y) ** 2)
            )
            S[j, i] = S[i, j] = var / (4 * N)

    # Return DataFrame with results
    cmas_pd = pd.DataFrame({
        'CMA': cmas,
        'SD': np.sqrt(np.diag(S)),
        'P(H0: CMA=0.5)': phalf
    })

    return cmas, S, cmas_pd




def one_dim_test(y_rank, x_rank):
    N = len(y_rank)


    zeta_3Y = zeta_fun(y_rank)
    k_zeta = prob_y(y_rank)**2 - zeta_3Y
    sigma_zeta = 9*np.mean(k_zeta**2)

    rho, cmas = comp_rho_cma(y_rank, x_rank)

    factor = 1 / ((1-zeta_3Y)**2)

    y_rank_unique, y_num = np.unique(y_rank, return_counts=True)
    pos_y = np.insert(np.cumsum(y_num), 0, 0)
    
    # Sort based on y_rank
    ix_y = np.argsort(y_rank)
    y_rank_sort = y_rank[ix_y]

    x_rank_sort = x_rank[ix_y]
    x_rank_unique = np.unique(x_rank_sort)
    k_p =  kernel_ties_optim2(x_rank_sort, y_rank_sort, rho, y_rank_unique, y_num, pos_y, ix_y, x_rank_unique)
    sigma_rho = 9*np.mean(k_p**2)
    sigma_pz = 9*np.mean((k_p * k_zeta)) 
    var = factor*(sigma_rho + (2*rho*sigma_pz)/(1-zeta_3Y) + (rho**2*sigma_zeta)/((1-zeta_3Y)**2))
    sd_2 = var/(4*N)
    phalf = 1 - norm.cdf((cmas - 0.5) / np.sqrt(var/(4*N)))

    return cmas, np.sqrt(sd_2), phalf


class test_multiple(object):

    def __init__(self, cmas, differences, covariance, global_p, global_z):
        self.cmas = cmas
        self.differences = differences
        self.covariance = covariance
        self.global_z = global_z
        self.global_p = global_p


    def print(self):
        print('CMA test: \n', self.cmas)
        print('\n')
        print('Pairwise test: \n', self.differences)
        print('\n')
        print('Covariance: \n', self.covariance)
        print('\n')
        print('Global z-value: ' , self.global_z)
        print('Global p-value: ' ,self.global_p)

class test_one(object):

    def __init__(self, cmas, sd, p):
        self.cmas = cmas
        self.sd = sd
        self.p = p

    def print(self):
        output_pd = pd.DataFrame({'CMA': [self.cmas], 'SD': [self.sd], 'P(H0: CMA = 0.5)': self.p})
        print('CMA test: \n', output_pd)


def calc_pvalue_chi_our(aucs, S):
    nauc = len(aucs)

    # Initialize L matrix with zeros
    L = np.zeros((nauc*(nauc-1)//2, nauc))

    newa = 0
    for i in range(nauc-1):
        newl = nauc - (i + 1)
    
        # Assign 1 to the first part of the slice
        L[newa:(newa+newl), i] = np.ones(newl)
    
        # Assign diagonal -1 values to the next part of the slice
        L[newa:(newa+newl), (i+1):(i+1+newl)] = -np.eye(newl)
    
        newa += newl


    aucdiff = L @ aucs
    L_S_Lt = L @ S @ L.T 
    # use R function from rms library matinv
    
    #numpy2ri.activate()
    #rms = importr('rms')
    # Convert the numpy array to an R object
    #r_matrix = ro.r['as.matrix'](L_S_Lt)
    
    #L_S_Lt_inv = np.array(rms.matinv(r_matrix))
    L_S_Lt_inv = np.linalg.pinv(L_S_Lt, rcond=1e-12)
    z = aucdiff.T @ L_S_Lt_inv @ aucdiff
    
    # Compute degrees of freedom using rank of the matrix
    _, R = np.linalg.qr(L_S_Lt)  # QR decomposition to get rank
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)  # Count non-zero diagonal elements

    # Calculate p-value using chi-squared distribution
    p = chi2.sf(z, df=rank) 
    return z, p, aucdiff




def pairwise_testing_our(nauc, S, aucdiff, conf_level):
    cor_auc = np.zeros((nauc * (nauc - 1)) // 2)
    ci = np.zeros(((nauc * (nauc - 1)) // 2, 2))
    pairp = np.zeros((nauc * (nauc - 1)) // 2)
    rows = []
    ctr = 0
    quantil = norm.ppf(1 - (1 - conf_level) / 2)
    #numpy2ri.activate()
    #rms = importr('rms')
    # Loop through pairs of AUCs
    for i in range(nauc - 1):
        for j in range(i + 1, nauc):
            cor_auc[ctr] = S[i, j] / np.sqrt(S[i, i] * S[j, j])
        
            # Compute LSL
            LSL = np.dot(np.dot(np.array([1, -1]), S[[j, i], :][:, [j, i]]), np.array([1, -1]))
    
            # Compute tmpz and pairp
            tmpz = aucdiff[ctr] / np.sqrt(LSL)
            pairp[ctr] = chi2.sf(aucdiff[ctr]**2 / LSL, df=1)
        
            # Compute confidence interval
            ci[ctr, 0] = aucdiff[ctr] - quantil * np.sqrt(LSL)
            ci[ctr, 1] = aucdiff[ctr] + quantil * np.sqrt(LSL)
        
            # Track row names (i vs j)
            rows.append(f"{i+1} vs. {j+1}")
            ctr += 1
        
    return pd.DataFrame({'Test': rows, 'CMA diff': aucdiff, 'CI(lower)': ci[:, 0] ,'CI(upper)': ci[:, 1], 'p.value': pairp,'correlation':cor_auc})



def agc_stat_test(y, x ,conf_level = 0.95):
    ##################
    # checking N > 3 #
    ##################
    
    # single prediction for y:
    if x.ndim == 1:
        # computation performed on ranks:
        y_ranks = rankdata(y, method='average')
        x_ranks = rankdata(x, method='average')
        
        # compute score, variance estimation and p-value
        cmas, sd, p = one_dim_test(y_ranks, x_ranks)
        
        return test_one(cmas = cmas, sd = sd, p = p)
    
    # multiple predictors for y:
    else:
        # computation performed on ranks:
        y_ranks = rankdata(y, method='average')
        xarray_ranks = np.apply_along_axis(rankdata, axis=1, arr=x, method='average')

        # single value testing and variance estimation: 
        cmas, S , cma_pd = Sigma_fast2(y_ranks, xarray_ranks)

        # global testing:
        z, p, cmadiff = calc_pvalue_chi_our(cmas, S)

        # pairwise testing:
        diff_pd = pairwise_testing_our(len(cmas), S, cmadiff, conf_level)
    
        return test_multiple(cmas = cma_pd, differences = diff_pd, covariance = S, global_z = z, global_p = p)



def meng_test_corr(r1, r2, r12, n, alternative="two.sided"):
    """
    Compute Meng, Rosenthal & Rubin (1992) test for comparing two dependent correlations
    with overlapping variables.
    """
    # Fisher's z-transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    
    # Following the exact cocor implementation from Meng et al. (1992)
    r_squared_avg = 0.5 * (r1**2 + r2**2)  # This is (r1² + r2²)/2
    f = (1 - r12) / (2 * (1 - r_squared_avg))
    
    # IMPORTANT: Add the f constraint from cocor package (f must be ≤ 1)
    if not np.isnan(f) and f > 1:
        f = 1
    
    # h = 1 + (r1² + r2²)/2 / (1 - (r1² + r2²)/2) * (1-f)
    h = 1 + r_squared_avg / (1 - r_squared_avg) * (1 - f)
    
    # Test statistic: z = (z1 - z2) * sqrt((n-3) / (2*(1-r12)*h))
    z_stat = (z1 - z2) * np.sqrt((n - 3) / (2 * (1 - r12) * h))
    
    # Compute p-value for two-sided test
    if alternative == "two.sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "one.sided":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        raise ValueError(f"Invalid alternative: '{alternative}'. Must be 'two.sided' or 'one.sided'.")
    return p_value


def meng_stat_test_spear(Y, X, alternative="two.sided"):
    """Meng test wrapper using agc correlation"""
    X1 = X[0, :]
    X2 = X[1, :]
    
    #r1 = spear(Y, X1)
    #r2 = spear(Y, X2)
    #r12 = spear(X1, X2)

    r1, p = spearmanr(Y, X1)
    r2, p = spearmanr(Y, X2)
    r12, p = spearmanr(X1, X2)
    
    n = len(Y)
    p_value = meng_test_corr(r1, r2, r12, n, alternative = alternative)
    
    return p_value
    

def run_simulation_meng_our(n, T = 10000, discrete = False, alternative="two.sided"):
    sigma_2 = sigma_1 = 1
    
    ps_meng = []
    ps_our = []
    if discrete:
        for i in tqdm(range(T), desc=f"n={n}"):
            X_0 = np.random.normal(0, 1, n)
            Z_1 = np.random.normal(0, sigma_1, n)
            Z_2 = np.random.normal(0, sigma_2, n)
            Y_0 = np.round(np.random.normal(X_0, 1, n))
            X_1 =np.round(X_0 + Z_1)
            X_2 = np.round(X_0 + Z_2)
        
            # Meng test (correct for dependent overlapping correlations)
            test_obj = agc_stat_test(Y_0, np.vstack((X_1, X_2)))
            p_val_our = test_obj.global_p
            p_val_meng = meng_stat_test_spear(Y_0, np.vstack((X_1, X_2)), alternative = alternative)
            ps_meng.append(p_val_meng) 
            ps_our.append(p_val_our)
    else:
        for i in tqdm(range(T), desc=f"n={n}"):
            X_0 = np.random.normal(0, 1, n)
            Z_1 = np.random.normal(0, sigma_1, n)
            Z_2 = np.random.normal(0, sigma_2, n)
            Y_0 = np.random.normal(X_0, 1, n)
            X_1 =X_0 + Z_1
            X_2 = X_0 + Z_2
        
            # Meng test (correct for dependent overlapping correlations)
            test_obj = agc_stat_test(Y_0, np.vstack((X_1, X_2)))
            p_val_our = test_obj.global_p
            p_val_meng = meng_stat_test_spear(Y_0, np.vstack((X_1, X_2)), alternative = alternative)
            ps_meng.append(p_val_meng)
            ps_our.append(p_val_our)

    return ps_meng, ps_our








    
    
