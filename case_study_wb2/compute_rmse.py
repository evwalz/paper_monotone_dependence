import xarray as xr
import numpy as np
import os
import pandas as pd
import sys
import argparse
import time
import glob

def rmse(forecast, obs):
    """Compute RMSE between forecast and observations."""
    # Compute squared differences
    squared_diff = (forecast - obs) ** 2
    
    # Mean over longitude and time
    mse_per_lat = squared_diff.mean(dim=['longitude', 'time'])
    
    # Take square root to get RMSE
    rmse_per_lat = np.sqrt(mse_per_lat)
    
    return rmse_per_lat.values

def _compute_rmse_grid(var_forecast, var_obs, start_time):
    """Helper function to compute RMSE per latitude."""
    # Get spatial dimension names
    lat_name = None
    lon_name = None
    
    for candidate in ['latitude', 'lat']:
        if candidate in var_forecast.dims:
            lat_name = candidate
            break
    
    for candidate in ['longitude', 'lon']:
        if candidate in var_forecast.dims:
            lon_name = candidate
            break
    
    if lat_name is None or lon_name is None:
        raise ValueError(f"Spatial dimensions not found. Available dims: {list(var_forecast.dims)}")
    
    latitudes = var_forecast[lat_name].values

    
    # Compute squared differences
    squared_diff = (var_forecast - var_obs) ** 2
    
    # Mean over longitude and time dimensions
    mse_per_lat = squared_diff.mean(dim=[lon_name, 'time'])
    
    # Take square root to get RMSE per latitude
    rmse_per_lat = np.sqrt(mse_per_lat)
    
    total_time = time.time() - start_time
    print(f"RMSE computation complete in {total_time:.2f} seconds.")
    
    return rmse_per_lat.values

def standardize_dims(dataset: xr.Dataset) -> xr.Dataset:
    """Standardize dimension and coordinate names to time, latitude, longitude."""
    dim_mapping = {}
    
    if 'time' not in dataset.dims:
        time_candidates = [dim for dim in dataset.dims if 'time' in dim.lower()]
        if time_candidates:
            dim_mapping[time_candidates[0]] = 'time'
    
    if 'latitude' not in dataset.dims:
        if 'lat' in dataset.dims:
            dim_mapping['lat'] = 'latitude'
    
    if 'longitude' not in dataset.dims:
        if 'lon' in dataset.dims:
            dim_mapping['lon'] = 'longitude'
    
    coord_mapping = {}
    if 'latitude' not in dataset.coords:
        if 'lat' in dataset.coords:
            coord_mapping['lat'] = 'latitude'
    
    if 'longitude' not in dataset.coords:
        if 'lon' in dataset.coords:
            coord_mapping['lon'] = 'longitude'
    
    if dim_mapping:
        dataset = dataset.rename(dim_mapping)
    
    if coord_mapping and coord_mapping != dim_mapping:
        dataset = dataset.rename(coord_mapping)
    
    return dataset

def make_latitude_increasing(dataset: xr.Dataset) -> xr.Dataset:
    """Make sure latitude values are increasing. Flip dataset if necessary."""
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

def get_model_name_from_path(file_path):
    """Extract model name from file path."""
    filename = os.path.basename(file_path)
    
    if filename.startswith('persistence_'):
        return 'persistence'
    elif filename.startswith('graphcast_'):
        return 'graphcast'
    elif filename.startswith('ifs_hres_'):
        return 'ifs_hres'
    elif filename.startswith('pangu_'):
        return 'pangu'
    elif filename.startswith('clim_'):
        return 'climatology'
    else:
        return filename.split('_')[0]

def get_variable_from_path(file_path):
    """Extract variable name from file path with improved logic."""
    filename = os.path.basename(file_path)
    name_without_ext = filename.replace('.zarr', '')
    
    known_variables = [
        'total_precipitation_24hr',
        '2m_temperature', 
        '10m_wind_speed',
        'mean_sea_level_pressure'
    ]
    
    for var in known_variables:
        if var in filename:
            return var
    
    parts = name_without_ext.split('_')
    model_prefixes = ['graphcast', 'ifs', 'hres', 'era5', 'obs', 'clim', 'persistence', 'pangu']
    year_suffix = '2020'
    
    start_idx = 0
    end_idx = len(parts)
    
    for i, part in enumerate(parts):
        if part not in model_prefixes:
            start_idx = i
            break
    
    for i in range(len(parts)-1, -1, -1):
        if parts[i] == year_suffix or parts[i].endswith('h'):
            end_idx = i
            break
    
    if start_idx < end_idx:
        variable_parts = parts[start_idx:end_idx]
        return '_'.join(variable_parts)
    
    return 'unknown_variable'

def rmse_per_lat(forecast, obs, variable, lead_time_hours):
    """
    Compute RMSE per latitude using the reference approach:
    1. Shift forecast time coordinates by lead_time (init time -> valid time)
    2. Align forecast and observations on valid time
    3. Compute RMSE
    """
    start_time = time.time()
    
    # Step 1: Shift forecast time coordinates to valid time
    lead_time_delta = np.timedelta64(lead_time_hours, 'h')
    forecast_valid = forecast.assign_coords(time=forecast.time + lead_time_delta)
    
    # Step 2: Extract the variable
    var_forecast = forecast_valid[variable]
    var_obs = obs[variable]
    
    # Step 3: Align on common valid times
    common_times = np.intersect1d(var_forecast.time, var_obs.time)
    var_forecast = var_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)

    
    return _compute_rmse_grid(var_forecast, var_obs, start_time)

def clim_rmse(climatology, obs, forecast_init_times, variable, lead_time_hours):
    """
    Compute RMSE for climatology using the reference approach.
    """
    start_time = time.time()
    
    # Calculate valid times from forecast initialization times
    lead_time_delta = np.timedelta64(lead_time_hours, 'h')
    valid_times = forecast_init_times + lead_time_delta
    
    # Filter to times that exist in observations
    valid_times = [t for t in valid_times if t in obs.time.values]
    
    # Get observations at valid times
    obs_valid = obs.sel(time=valid_times)
    
    # Create a time selection dictionary for climatology
    time_selection = {
        'dayofyear': xr.DataArray([pd.Timestamp(t).dayofyear for t in valid_times], dims='time')
    }
    
    # Add hour selection if climatology includes hourly data
    if 'hour' in climatology.coords:
        time_selection['hour'] = xr.DataArray([pd.Timestamp(t).hour for t in valid_times], dims='time')

    # Select climatology values based on the time selection
    clim_forecast = climatology[variable].sel(time_selection)
    
    # Assign the valid times as coordinates to climatology
    clim_forecast = clim_forecast.assign_coords(time=valid_times)
    
    # Get the observations for the same variable
    var_obs = obs_valid[variable]
    
    # Make sure time dimensions match
    common_times = np.intersect1d(clim_forecast.time, var_obs.time)
    var_forecast = clim_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)


    return _compute_rmse_grid(var_forecast, var_obs, start_time)

def compute_rmse(input_dir, variable=None, lead_times=[24, 48, 72]):
    """Compute RMSE for all forecast models for a specific variable at multiple lead times."""
    output_dir = os.path.join(input_dir, 'rmse_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # If variable not specified, try to auto-detect from available files
    if variable is None:
        obs_files = glob.glob(os.path.join(input_dir, 'era5_obs_*.zarr'))
        if not obs_files:
            raise ValueError("No ERA5 observation files found. Please specify variable explicitly.")
        
        obs_path = obs_files[0]
        variable = get_variable_from_path(obs_path)
        print(f"Auto-detected variable: {variable}")
    else:
        obs_path = os.path.join(input_dir, f'era5_obs_{variable}_2020.zarr')
    
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Observation file not found: {obs_path}")
    
    #print(f"Loading observations from: {obs_path}")
    obs = open_obs(obs_path)
    
    # Loop over each lead time
    for lead_time_hours in lead_times:
        
        # Find all forecast files for this variable and lead time
        forecast_patterns = [
            f'graphcast_ifs_{variable}_{lead_time_hours}h_2020.zarr',
            f'ifs_hres_{variable}_{lead_time_hours}h_2020.zarr', 
            f'persistence_{variable}_{lead_time_hours}h_2020.zarr'
        ]
        
        forecast_init_times = None
        
        for pattern in forecast_patterns:
            fct_path = os.path.join(input_dir, pattern)
            if os.path.exists(fct_path):
                print(f"\nProcessing forecast: {pattern}")
                forecast = open_fct(fct_path)
                
                # Store forecast initialization times for climatology processing
                if forecast_init_times is None:
                    forecast_init_times = forecast.time.values
                
                rmse_vals = rmse_per_lat(forecast, obs, variable, lead_time_hours)
                model_name = get_model_name_from_path(pattern)
                
                # Save results
                rmse_file = os.path.join(output_dir, f'rmse_{model_name}_{variable}_lead{lead_time_hours}h.txt')
                
                np.savetxt(rmse_file, rmse_vals)
                
                print(f"Saved RMSE results to: {rmse_file}")
            else:
                print(f"Warning: Forecast file not found: {fct_path}")

        # Process climatology for this lead time
        clim_path = os.path.join(input_dir, f'clim_{variable}_{lead_time_hours}h_2020.zarr')
        if os.path.exists(clim_path):
            #print(f"\nProcessing climatology: {clim_path} at {lead_time_hours}h lead time")
            climatology = open_fct(clim_path)
            
            if forecast_init_times is None:
                raise ValueError("No forecast files were processed successfully. Cannot determine forecast times for climatology.")
            
            rmse_vals = clim_rmse(climatology, obs, forecast_init_times, variable, lead_time_hours)
            
            # Save climatology results
            rmse_file = os.path.join(output_dir, f'rmse_climatology_{variable}_lead{lead_time_hours}h.txt')
            
            np.savetxt(rmse_file, rmse_vals)
            
            print(f"Saved climatology RMSE results to: {rmse_file}")
        else:
            print(f"Warning: Climatology file not found: {clim_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute RMSE values for weather forecasts at multiple lead times')
    parser.add_argument('--input_dir', type=str, default='./fct_data/', 
                       help='Input directory containing the forecast and observation data')
    parser.add_argument('--variable', type=str, default='total_precipitation_24hr',
                       help='Variable to process. Options: 2m_temperature, 10m_wind_speed, mean_sea_level_pressure, total_precipitation_24hr.')
    parser.add_argument('--lead_times', type=int, nargs='+', default=[24],
                       help='Lead times in hours to compute RMSE for')
    
    args = parser.parse_args()
    
    # Ensure input directory exists and has proper format
    input_dir = args.input_dir.rstrip('/')
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    #print(f"Processing data from: {input_dir}")
    #print(f"Lead times: {args.lead_times}")
    compute_rmse(input_dir, args.variable, args.lead_times)
    return 0

if __name__ == "__main__":
    sys.exit(main())
