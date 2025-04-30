import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

forecast_models = {
    'GraphCast': 'graphcast_ifs',
    'IFS Mean': 'ifs_mean',
    'IFS HRES': 'ifs_hres',
    'Persistence': 'persistence', 
    'Climatology': 'climatology'
}

import seaborn as sns
color_codes = sns.color_palette("colorblind", 6)
model_color_map = {
        'Persistence': '#9467bd',##color_codes[4], # 
        'Climatology': color_codes[3],
        'GraphCast': color_codes[0],
        'IFS Mean': color_codes[1],
        'IFS HRES': color_codes[2]
    }

def get_rmse(input_dir):
    rmse_per_lat_results = {}

    forecast_models = {
    'GraphCast':  'rmse_graphcast_ifs.txt',
    'IFS Mean': 'rmse_ifs_mean.txt',
    'IFS HRES': 'rmse_ifs_hres.txt',
    'Climatology': 'rmse_climatology.txt',
    'Persistence': 'rmse_persistence.txt'
    }

    for model_name, forecast_path in forecast_models.items():
        rmse_per_lat = np.loadtxt(input_dir + forecast_path)
        rmse_per_lat_results[model_name] = rmse_per_lat

    return rmse_per_lat_results

def get_acc(input_dir):
    acc_per_lat_results = {}

    forecast_models = {
    'GraphCast':  'acc_graphcast_ifs.txt',
    'IFS Mean': 'acc_ifs_mean.txt',
    'IFS HRES': 'acc_ifs_hres.txt',
    'Climatology': 'acc_climatology.txt',
    'Persistence': 'acc_persistence.txt'
    }

    for model_name, forecast_path in forecast_models.items():
        acc_per_lat = np.loadtxt(input_dir + forecast_path)
        acc_per_lat_results[model_name] = acc_per_lat

    return acc_per_lat_results

def get_seeps(input_dir):
    seeps_per_lat_results = {}

    forecast_models = {
    'GraphCast':  'seeps_graphcast_ifs.nc',
    'IFS Mean': 'seeps_ifs_mean.nc',
    'IFS HRES': 'seeps_ifs_hres.nc',
    'Climatology': 'seeps_climatology.nc',
    'Persistence': 'seeps_persistence.nc'
    }

    for model_name, forecast_path in forecast_models.items():
        seeps_per_lat = xr.open_dataset(input_dir + forecast_path)
        seeps_per_lat_results[model_name] = seeps_per_lat.total_precipitation_24hr.values

    return seeps_per_lat_results

def get_cma(input_dir):
    cma_cpa_per_lat_results = {}

    forecast_models = {
    'GraphCast':  'cma_graphcast_ifs.txt',
    'IFS Mean': 'cma_ifs_mean.txt',
    'IFS HRES': 'cma_ifs_hres.txt',
    'Persistence': 'cma_persistence.txt',
    'Climatology': 'cma_climatology.txt'
    }

    for model_name, forecast_path in forecast_models.items():
        cma_cpa_per_lat = np.loadtxt(input_dir + forecast_path)
        cma_cpa_per_lat_results[model_name] = cma_cpa_per_lat

    return cma_cpa_per_lat_results


def get_cpa(input_dir):
    cma_cpa_per_lat_results = {}

    forecast_models = {
    'GraphCast':  'cpa_graphcast_ifs.txt',
    'IFS Mean': 'cpa_ifs_mean.txt',
    'IFS HRES': 'cpa_ifs_hres.txt',
    'Climatology': 'cpa_climatology.txt',
    'Persistence': 'cpa_persistence.txt'
    }

    for model_name, forecast_path in forecast_models.items():
        cma_cpa_per_lat = np.loadtxt(input_dir + forecast_path)
        cma_cpa_per_lat_results[model_name] = cma_cpa_per_lat

    return cma_cpa_per_lat_results

def calculate_improvement_over_climatology(rmse_per_lat_results):
    """
    Calculate improvement percentage of each model over climatology per latitude
    
    Args:
        rmse_per_lat_results: Dictionary containing model name to RMSE per latitude mapping
        
    Returns:
        Dictionary containing model name to improvement percentage per latitude mapping
    """
    climatology_rmse = rmse_per_lat_results['Climatology']
    improvement_results = {}
    
    for model_name, rmse_per_lat in rmse_per_lat_results.items():
        #if model_name != 'Climatology':
            # Calculate improvement percentage: (climatology_rmse - model_rmse) / climatology_rmse * 100
        improvement = (climatology_rmse - rmse_per_lat) / climatology_rmse * 100
        improvement_results[model_name] = improvement
    
    return improvement_results

def calculate_improvement_over_climatology_corr(rmse_per_lat_results):
    """
    Calculate improvement percentage of each model over climatology per latitude
    
    Args:
        rmse_per_lat_results: Dictionary containing model name to RMSE per latitude mapping
        
    Returns:
        Dictionary containing model name to improvement percentage per latitude mapping
    """
    climatology_rmse = rmse_per_lat_results['Climatology']
    improvement_results = {}
    
    for model_name, rmse_per_lat in rmse_per_lat_results.items():
        #if model_name != 'Climatology':
            # Calculate improvement percentage: (climatology_rmse - model_rmse) / climatology_rmse * 100
        improvement = (rmse_per_lat - climatology_rmse) / climatology_rmse * 100
        improvement_results[model_name] = improvement
    
    return improvement_results





