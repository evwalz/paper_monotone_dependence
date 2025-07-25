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

    

def acc_per_lat(forecast, obs, climatology):

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
    clim_var = climatology[var]
        
    # Make sure the time dimensions match

    var_obs = var_obs.sel(time=slice(var_forecast.time.min(), var_forecast.time.max()))

    # Handle potentially missing values
    common_times = np.intersect1d(var_forecast.time, var_obs.time)
    var_forecast = var_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)

    time_selection = dict(dayofyear=var_forecast.time.dt.dayofyear)
    if "hour" in climatology.coords:
        time_selection["hour"] = var_forecast.time.dt.hour
    
    clim_for_times = clim_var.sel(time_selection).load()
    
    # Compute anomalies
    forecast_anom = var_forecast - clim_for_times
    truth_anom = var_obs - clim_for_times

    numerator = (forecast_anom * truth_anom).mean("longitude", skipna=True)
    forecast_std = np.sqrt((forecast_anom ** 2).mean("longitude", skipna=True))
    truth_std = np.sqrt((truth_anom ** 2).mean("longitude", skipna=True))
    acc = numerator / (forecast_std * truth_std)
    acc_lats = acc.mean(dim = 'time')
        
    # numerator = forecast_anom * truth_anom
    # forecast_std = np.sqrt(forecast_anom ** 2)
    # truth_std = np.sqrt(truth_anom ** 2)
    # acc = numerator / (forecast_std * truth_std)       
    # acc_lats = acc.mean(dim=['longitude', 'time'])
    
    return acc_lats



def compute_acc(input_dir):
    output_dir = input_dir + '/acc_results/'
    os.makedirs(output_dir, exist_ok = True)
    
    obs_path = input_dir + 'era5_obs_total_precipitation_24hr_2020.zarr'
    obs = open_obs(obs_path)

    climatology = open_fct(input_dir + 'clim_total_precipitation_24hr_2020.zarr')
    
    forecast_paths = ['graphcast_ifs_total_precipitation_24hr_2020.zarr', 'ifs_hres_total_precipitation_24hr_2020.zarr', 'persistence_total_precipitation_24hr_2020.zarr']
    names = ['graphcast_ifs', 'ifs_hres', 'persistence']

    for fct_path, save_name in zip(forecast_paths, names):
        forecast = open_fct(input_dir + fct_path)
        acc_vals = acc_per_lat(forecast, obs, climatology)
        np.savetxt(output_dir + 'acc_'+save_name+'.txt', acc_vals)

    # compute for climatology:
    acc_vals = np.zeros_like(acc_vals, dtype=np.float64)
    np.savetxt(output_dir + 'acc_climatology.txt', acc_vals)
  


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_dir', type=str, default='./fct_data/', help='Output file path')
    args = parser.parse_args()
    input_dir = args.input_dir
    compute_acc(input_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())

