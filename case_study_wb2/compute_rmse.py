import xarray as xr
import numpy as np
import os
import pandas as pd
import sys
import argparse


def clim_rmse(climatology, obs, time_fct):
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
    clim_forecast = climatology[var].sel(time_selection)
    
    # Assign the target times as coordinates to climatology
    clim_forecast = clim_forecast.assign_coords(time=valid_target_times)
    # Get the observations for the same variable
    var_obs = ground_truth[var]
    
    # Make sure time dimensions match
    common_times = np.intersect1d(clim_forecast.time, var_obs.time)
    clim_forecast = clim_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)
    
    # Compute RMSE
    squared_diff = (clim_forecast - var_obs) ** 2

    mse_per_lat = squared_diff.mean(dim=['longitude', 'time'])
    
    # Take square root to get RMSE per latitude
    rmse_per_lat = np.sqrt(mse_per_lat)
    
    return rmse_per_lat


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
    

def rmse_per_lat(forecast, obs):

    forecast_times = forecast.time.values
    obs_times = obs.time.values

    # Calculate the times in observations that are 24h after each forecast time
    #target_times = forecast_times + np.timedelta64(24, 'h')
    # Filter to only include forecast times that have a corresponding obs time 24h later
    valid_forecast_times = [t for t in forecast_times if t + np.timedelta64(24, 'h') in obs_times]
    valid_obs_times = valid_forecast_times+ np.timedelta64(24, 'h')
    # Select only those valid forecast times
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
        
    # Compute RMSE
    squared_diff = (var_forecast - var_obs) ** 2

    mse_per_lat = squared_diff.mean(dim=['longitude', 'time'])
    
    # Take square root to get RMSE per latitude
    rmse_per_lat = np.sqrt(mse_per_lat)
    
    return rmse_per_lat



def compute_rmse(input_dir):
    output_dir = input_dir + '/rmse_results/'
    os.makedirs(output_dir, exist_ok = True)
    
    obs_path = input_dir + 'era5_obs_precipitation_24hr_2020.zarr'
    obs = open_obs(obs_path)
    
    forecast_paths = ['graphcast_ifs_precipitation_24hr_2020.zarr', 'ifs_hres_precipitation_24hr_2020.zarr', 'ifs_mean_precipitation_24hr_2020.zarr', 'persistence_precipitation_24hr_2020.zarr']

    for fct_path in forecast_paths:
        forecast = open_fct(input_dir + fct_path)
        rmse_vals = rmse_per_lat(forecast, obs)
        save_name = get_letters_until_second_underscore(fct_path)
        np.savetxt(output_dir + 'rmse_'+save_name+'.txt', rmse_vals)

    time_fct = forecast.time.values

    # Climatology:
    climatology = open_fct(input_dir + 'clim_precipitation_24hr_2020.zarr')
    rmse_vals = clim_rmse(climatology, obs, time_fct)
    np.savetxt(output_dir + 'rmse_climatology.txt', rmse_vals)
    

    


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_dir', type=str, default='./fct_data/', help='Output file path')
    args = parser.parse_args()
    input_dir = args.input_dir
    compute_rmse(input_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())



#def compute_cpa():

#def compute_cma():

#def compute_seeps():

#def compute_acc():


