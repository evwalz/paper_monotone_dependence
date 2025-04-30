import xarray as xr
import numpy as np
import os
import pandas as pd
import sys
import argparse
from scipy.stats import rankdata
import time
    
    # Function to compute CMA - vectorized when possible
def cma_cpa(y, x):
    if len(y) <= 1 or len(x) <= 1:
        return np.nan, np.nan
            
        # Handle NaN values
    valid_idx = ~np.isnan(y) & ~np.isnan(x)
    if np.sum(valid_idx) <= 1:
        return np.nan, np.nan
            
    y_valid = y[valid_idx]
    x_valid = x[valid_idx]
        
    y_rank = rankdata(y_valid, method='average')
    x_rank = rankdata(x_valid, method='average')
    y_classes = rankdata(y_valid, method='dense')
    N = len(y_valid)
    var = np.sum((y_rank - np.mean(y_rank))**2)*(1/(N-1))
    var_cpa = np.cov(y_classes, y_rank)[0][1]
    return (np.cov(y_rank, x_rank)[0][1]/var+1)/2, (np.cov(y_classes, x_rank)[0][1]/var_cpa+1)/2


def clim_cma_cpa(climatology, obs, time_fct):
    obs_times = obs.time.values
    
    # Calculate the target verification times (t+24h)
    valid_times = []
    valid_target_times = []
    
    valid_forecast_times = [t for t in time_fct if t + np.timedelta64(24, 'h') in obs_times]
    valid_target_times = [t + np.timedelta64(24, 'h') for t in valid_forecast_times]
    
    # Get the ground truth (observations at time t+24h)
    ground_truth = obs.sel(time=valid_target_times)
    
    # Get climatology values for each target time
    var = 'total_precipitation_24hr'
    
    # Create a time selection dictionary
    time_selection = {
        'dayofyear': xr.DataArray([pd.Timestamp(t).dayofyear for t in valid_target_times], dims='time')
    }
    
    # Add hour selection if climatology includes hourly data
    if 'hour' in climatology.coords:
        time_selection['hour'] = xr.DataArray([pd.Timestamp(t).hour for t in valid_target_times], dims='time')

    # Select climatology values based on the time selection
    start_time = time.time()
    clim_forecast = climatology[var].sel(time_selection)
    
    # Assign the target times as coordinates to climatology
    clim_forecast = clim_forecast.assign_coords(time=valid_target_times)
    # Get the observations for the same variable
    var_obs = ground_truth[var]
    
    # Make sure time dimensions match
    common_times = np.intersect1d(clim_forecast.time, var_obs.time)
    var_forecast = clim_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)

        # Compute CMA and CPA
    # Get dimensions
    latitudes = var_forecast.latitude.values
    longitudes = var_forecast.longitude.values

    print(f"Data prepared in {time.time() - start_time:.2f} seconds")
    print(f"Computing CMA/CPA for {len(latitudes)} latitudes × {len(longitudes)} longitudes...")
    
    # Initialize array to store results
    cma_grid = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    cpa_grid = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    
    # Process in batches of latitudes
    batch_size = max(1, len(latitudes) // 40)  # Divide into ~20 batches
    
    for batch_start in range(0, len(latitudes), batch_size):
        batch_time = time.time()
        
        batch_end = min(batch_start + batch_size, len(latitudes))
        lat_batch = latitudes[batch_start:batch_end]
        
        # Extract data for this batch of latitudes
        batch_forecast = var_forecast.sel(latitude=lat_batch)
        batch_obs = var_obs.sel(latitude=lat_batch)
        
        # Process each latitude in the batch
        for i, lat in enumerate(lat_batch):
            # Get the current latitude index in the full array
            lat_idx = batch_start + i
            
            # Convert to numpy for faster processing
            fct_data = batch_forecast.sel(latitude=lat).values
            obs_data = batch_obs.sel(latitude=lat).values
            
            # Process each longitude in parallel if possible, otherwise sequentially
            for j, lon in enumerate(longitudes):
                # Extract time series for this lat-lon point
                fct_series = fct_data[:, j]  # time series for this location
                obs_series = obs_data[:, j]
                
                # Compute CMA
                try:
                    cma_grid[lat_idx, j], cpa_grid[lat_idx, j] = cma_cpa(obs_series, fct_series)
                except Exception as e:
                    cma_grid[lat_idx, j] = np.nan 
                    cpa_grid[lat_idx, j] = np.nan
        
        # Report progress after each batch
        progress = (batch_end / len(latitudes)) * 100
        batch_elapsed = time.time() - batch_time
        total_elapsed = time.time() - start_time
        remaining = (total_elapsed / progress) * (100 - progress) if progress > 0 else 0
        
        print(f"Progress: {progress:.1f}% - Processed latitudes {batch_start+1}-{batch_end}/{len(latitudes)} "
              f"in {batch_elapsed:.1f}s (Elapsed: {total_elapsed:.1f}s, Est. remaining: {remaining:.1f}s)")
    
    # Compute the average CMA over longitudes for each latitude
    print("Computing average CMA per latitude...")
    cma_per_lat_values = np.nanmean(cma_grid, axis=1)
    cpa_per_lat_values = np.nanmean(cpa_grid, axis=1)
    
    total_time = time.time() - start_time
    print(f"CMA computation complete in {total_time:.2f} seconds.")
    
    return cma_per_lat_values, cpa_per_lat_values
    



def standardize_dims(dataset: xr.Dataset) -> xr.Dataset:
    """Standardize dimension names to time, latitude, longitude."""
    dim_mapping = {}
    
    # Check and map time dimension
    if 'time' not in dataset.dims:
        time_candidates = [dim for dim in dataset.dims if 'time' in dim.lower()]
        if time_candidates:
            dim_mapping[time_candidates[0]] = 'time'
    
    # Check and map latitude dimension
    if 'latitude' not in dataset.dims:
        if 'lat' in dataset.dims:
            dim_mapping['lat'] = 'latitude'
    
    # Check and map longitude dimension
    if 'longitude' not in dataset.dims:
        if 'lon' in dataset.dims:
            dim_mapping['lon'] = 'longitude'
    
    if dim_mapping:
        return dataset.rename(dim_mapping)
    return dataset

def make_latitude_increasing(dataset: xr.Dataset) -> xr.Dataset:
    """Make sure latitude values are increasing. Flip dataset if necessary."""
    # Get the latitude dimension name (could be 'lat' or 'latitude')
    lat_name = next((dim for dim in ['latitude', 'lat'] if dim in dataset.dims), None)
    if lat_name is None:
        raise ValueError("No latitude dimension found in the dataset")
    
    lat = dataset[lat_name].values
    if (np.diff(lat) < 0).all():
        reverse_lat = lat[::-1]
        dataset = dataset.sel({lat_name: reverse_lat})
    return dataset

def open_obs(obs_path):
    obs = xr.open_zarr(obs_path)
    obs = standardize_dims(obs)
    obs = make_latitude_increasing(obs)
    return obs

def open_fct(forecast_path):
    forecast = xr.open_zarr(forecast_path)
    forecast = standardize_dims(forecast)
    forecast = make_latitude_increasing(forecast)
    return forecast

def get_letters_until_second_underscore(s):
    """
    Extract all characters from a string until the second underscore.
    
    Args:
        s (str): The input string
        
    Returns:
        str: All characters before the second underscore, or the entire string 
             if fewer than two underscores exist
    """

    if s == 'persistence_precipitation_24hr_2020.zarr':
        return 'persistence'
    # Find the position of the first underscore
    first_underscore_index = s.find('_')
    
    # If first underscore doesn't exist, return the entire string
    if first_underscore_index == -1:
        return s
    
    # Find the position of the second underscore (start searching after the first one)
    second_underscore_index = s.find('_', first_underscore_index + 1)

    
    # If second underscore exists, return everything before it
    if second_underscore_index != -1:
        return s[:second_underscore_index]
    # If only one underscore exists, return the entire string
    else:
        return s
    

def cma_cpa_per_lat(forecast, obs):

    forecast_times = forecast.time.values
    obs_times = obs.time.values

    # Calculate the times in observations that are 24h after each forecast time
    #target_times = forecast_times + np.timedelta64(24, 'h')
    # Filter to only include forecast times that have a corresponding obs time 24h later
    valid_forecast_times = [t for t in forecast_times if t + np.timedelta64(24, 'h') in obs_times]
    valid_obs_times = valid_forecast_times+ np.timedelta64(24, 'h')
    # Select only those valid forecast times
    start_time = time.time()
    filtered_forecast = forecast.sel(time=valid_forecast_times)
    filtered_obs = obs.sel(time = valid_obs_times)
    filtered_forecast = filtered_forecast.assign_coords(time=valid_obs_times)
    var = 'total_precipitation_24hr'
    var_forecast = filtered_forecast[var]
    var_obs = filtered_obs[var]
        
    # Make sure the time dimensions match

    var_obs = var_obs.sel(time=slice(var_forecast.time.min(), var_forecast.time.max()))

    # Handle potentially missing values
    common_times = np.intersect1d(var_forecast.time, var_obs.time)
    var_forecast = var_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)
        
    # Compute CMA and CPA
    # Get dimensions
    latitudes = var_forecast.latitude.values
    longitudes = var_forecast.longitude.values
    
    print(f"Data prepared in {time.time() - start_time:.2f} seconds")
    print(f"Computing CMA/CPA for {len(latitudes)} latitudes × {len(longitudes)} longitudes...")
    
    # Initialize array to store results
    cma_grid = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    cpa_grid = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    
    # Process in batches of latitudes
    batch_size = max(1, len(latitudes) // 40)  # Divide into ~20 batches
    
    for batch_start in range(0, len(latitudes), batch_size):
        batch_time = time.time()
        
        batch_end = min(batch_start + batch_size, len(latitudes))
        lat_batch = latitudes[batch_start:batch_end]
        
        # Extract data for this batch of latitudes
        batch_forecast = var_forecast.sel(latitude=lat_batch)
        batch_obs = var_obs.sel(latitude=lat_batch)
        
        # Process each latitude in the batch
        for i, lat in enumerate(lat_batch):
            # Get the current latitude index in the full array
            lat_idx = batch_start + i
            
            # Convert to numpy for faster processing
            fct_data = batch_forecast.sel(latitude=lat).values
            obs_data = batch_obs.sel(latitude=lat).values
            
            # Process each longitude in parallel if possible, otherwise sequentially
            for j, lon in enumerate(longitudes):
                # Extract time series for this lat-lon point
                fct_series = fct_data[:, j]  # time series for this location
                obs_series = obs_data[:, j]
                
                # Compute CMA
                try:
                    cma_grid[lat_idx, j], cpa_grid[lat_idx, j] = compute_cma_cpa(obs_series, fct_series)
                except Exception as e:
                    print(f"Error at lat={lat}, lon={longitudes[j]}: {e}")
                    cma_grid[lat_idx, j] = np.nan 
                    cpa_grid[lat_idx, j] = np.nan
        
        # Report progress after each batch
        progress = (batch_end / len(latitudes)) * 100
        batch_elapsed = time.time() - batch_time
        total_elapsed = time.time() - start_time
        remaining = (total_elapsed / progress) * (100 - progress) if progress > 0 else 0
        
        print(f"Progress: {progress:.1f}% - Processed latitudes {batch_start+1}-{batch_end}/{len(latitudes)} "
              f"in {batch_elapsed:.1f}s (Elapsed: {total_elapsed:.1f}s, Est. remaining: {remaining:.1f}s)")
    
    # Compute the average CMA over longitudes for each latitude
    print("Computing average CMA per latitude...")
    cma_per_lat_values = np.nanmean(cma_grid, axis=1)
    cpa_per_lat_values = np.nanmean(cpa_grid, axis=1)
    
    total_time = time.time() - start_time
    print(f"CMA computation complete in {total_time:.2f} seconds.")
    
    return cma_per_lat_values, cpa_per_lat_values


def compute_cma_cpa(input_dir):
    output_dir = input_dir + '/cma_cpa_results/'
    os.makedirs(output_dir, exist_ok = True)
    
    obs_path = input_dir + 'era5_obs_precipitation_24hr_2020.zarr'
    obs = open_obs(obs_path)
    
    forecast_paths = ['graphcast_ifs_precipitation_24hr_2020.zarr', 'ifs_hres_precipitation_24hr_2020.zarr', 'ifs_mean_precipitation_24hr_2020.zarr', 'persistence_precipitation_24hr_2020.zarr']

    for fct_path in forecast_paths:
        forecast = open_fct(input_dir + fct_path)
        cma_vals, cpa_vals = cma_cpa_per_lat(forecast, obs)
        save_name = get_letters_until_second_underscore(fct_path)
        np.savetxt(output_dir + 'cma_'+save_name+'.txt', cma_vals)
        np.savetxt(output_dir + 'cpa_'+save_name+'.txt', cpa_vals)

    time_fct = forecast.time.values

    # Climatology:
    climatology = open_fct(input_dir + 'clim_precipitation_24hr_2020.zarr')
    cma_vals, cpa_vals = clim_cma_cpa(climatology, obs, time_fct)
    np.savetxt(output_dir + 'cma_climatology.txt', cma_vals)
    np.savetxt(output_dir + 'cpa_climatology.txt', cpa_vals)
    

    


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_dir', type=str, default='./fct_data/', help='Output file path')
    args = parser.parse_args()
    input_dir = args.input_dir
    compute_cma_cpa(input_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())