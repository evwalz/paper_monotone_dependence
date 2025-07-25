import xarray as xr
import numpy as np
import os
import pandas as pd
import sys
import argparse
from scipy.stats import rankdata
import time
    
    # Function to compute CMA - vectorized when possible
def cma(y, x):
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
    return (np.cov(y_rank, x_rank)[0][1]/var+1)/2


def clim_cma(climatology, obs, time_fct, variable):
    """Compute CMA for climatology - same logic as precipitation."""
    obs_times = obs.time.values
    
    # Calculate the target verification times (t+24h for timedelta_idx=3)
    valid_forecast_times = [t for t in time_fct if t + np.timedelta64(24, 'h') in obs_times]
    valid_target_times = [t + np.timedelta64(24, 'h') for t in valid_forecast_times]
    
    # Get the ground truth (observations at time t+12h)
    ground_truth = obs.sel(time=valid_target_times)
    
    # Create a time selection dictionary for climatology
    time_selection = {
        'dayofyear': xr.DataArray([pd.Timestamp(t).dayofyear for t in valid_target_times], dims='time')
    }
    
    # Add hour selection if climatology includes hourly data
    if 'hour' in climatology.coords:
        time_selection['hour'] = xr.DataArray([pd.Timestamp(t).hour for t in valid_target_times], dims='time')

    # Select climatology values based on the time selection
    start_time = time.time()
    clim_forecast = climatology[variable].sel(time_selection)
    
    # Assign the target times as coordinates to climatology
    clim_forecast = clim_forecast.assign_coords(time=valid_target_times)
    # Get the observations for the same variable
    var_obs = ground_truth[variable]
    
    # Make sure time dimensions match
    common_times = np.intersect1d(clim_forecast.time, var_obs.time)
    var_forecast = clim_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)

    return _compute_cma_grid(var_forecast, var_obs, start_time)

def _compute_cma_grid(var_forecast, var_obs, start_time):
    """Helper function to compute CMA on a grid."""
    # Get dimensions
    latitudes = var_forecast.latitude.values
    longitudes = var_forecast.longitude.values

    print(f"Data prepared in {time.time() - start_time:.2f} seconds")
    print(f"Computing CMA for {len(latitudes)} latitudes × {len(longitudes)} longitudes...")
    
    # Initialize array to store results
    cma_grid = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    
    # Process in batches of latitudes
    batch_size = max(1, len(latitudes) // 40)  # Divide into ~40 batches
    
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
            
            # Process each longitude
            for j, lon in enumerate(longitudes):
                # Extract time series for this lat-lon point
                fct_series = fct_data[:, j]  # time series for this location
                obs_series = obs_data[:, j]
                
                # Compute CMA
                try:
                    cma_grid[lat_idx, j] = cma(obs_series, fct_series)
                except Exception as e:
                    cma_grid[lat_idx, j] = np.nan
        
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
    
    total_time = time.time() - start_time
    print(f"CMA computation complete in {total_time:.2f} seconds.")
    
    return cma_per_lat_values
    


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

def get_model_name_from_path(file_path):
    """Extract model name from file path."""
    filename = os.path.basename(file_path)
    
    if filename.startswith('persistence_'):
        return 'persistence'
    elif filename.startswith('graphcast_'):
        return 'graphcast_ifs'
    elif filename.startswith('ifs_hres_'):
        return 'ifs_hres'
    elif filename.startswith('clim_'):
        return 'climatology'
    else:
        # Fallback: extract everything before the first underscore
        return filename.split('_')[0]

def get_variable_from_path(file_path):
    """Extract variable name from file path with improved logic."""
    filename = os.path.basename(file_path)
    
    # Remove the .zarr extension
    name_without_ext = filename.replace('.zarr', '')
    
    # Known variable patterns with their exact strings
    known_variables = [
        'total_precipitation_24hr',
        '2m_temperature', 
        '10m_wind_speed',
        'mean_sea_level_pressure'
    ]
    
    # Check for exact matches first
    for var in known_variables:
        if var in filename:
            return var
    
    # If no exact match, try to parse from structure
    # Expected format: {model}_{variable}_{year}.zarr
    parts = name_without_ext.split('_')
    
    # Remove known prefixes
    model_prefixes = ['graphcast', 'ifs', 'hres', 'era5', 'obs', 'clim', 'persistence']
    year_suffix = '2020'
    
    # Find the variable part(s) between model and year
    start_idx = 0
    end_idx = len(parts)
    
    # Skip model prefixes
    for i, part in enumerate(parts):
        if part not in model_prefixes:
            start_idx = i
            break
    
    # Find year suffix
    for i, part in enumerate(parts):
        if part == year_suffix:
            end_idx = i
            break
    
    if start_idx < end_idx:
        variable_parts = parts[start_idx:end_idx]
        return '_'.join(variable_parts)
    
    return 'unknown_variable'

def cma_per_lat(forecast, obs, variable):
    """Compute CMA per latitude for a specific variable - same logic as precipitation."""
    forecast_times = forecast.time.values
    obs_times = obs.time.values

    # Filter to only include forecast times that have a corresponding obs time 24h later
    # (using timedelta_idx=3 which corresponds to 24 hours in your dataset)
    valid_forecast_times = [t for t in forecast_times if t + np.timedelta64(24, 'h') in obs_times]
    valid_obs_times = np.array(valid_forecast_times) + np.timedelta64(24, 'h')
    
    # Select only those valid forecast times
    start_time = time.time()
    filtered_forecast = forecast.sel(time=valid_forecast_times)
    filtered_obs = obs.sel(time=valid_obs_times)
    
    # Align time coordinates for comparison
    filtered_forecast = filtered_forecast.assign_coords(time=valid_obs_times)
    
    var_forecast = filtered_forecast[variable]
    var_obs = filtered_obs[variable]
        
    # Handle potentially missing values
    common_times = np.intersect1d(var_forecast.time, var_obs.time)
    var_forecast = var_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)
    
    print(f"Variable: {variable}")
    print(f"Verification offset: 24 hours (timedelta_idx=3 in your dataset)")
    print(f"Number of verification times: {len(common_times)}")
        
    return _compute_cma_grid(var_forecast, var_obs, start_time)

def compute_cma(input_dir):
    """Compute CMA for all forecast models for a specific variable or auto-detect."""
    output_dir = os.path.join(input_dir, 'cma_results')
    os.makedirs(output_dir, exist_ok=True)
    variable = 'total_precipitation_24hr'
    
    # If variable not specified, try to auto-detect from available files
    obs_path = os.path.join(input_dir, f'era5_obs_{variable}_2020.zarr')
    obs = open_obs(obs_path)
    
    # Find all forecast files for this variable
    forecast_patterns = [
        f'graphcast_ifs_{variable}_2020.zarr',
        f'ifs_hres_{variable}_2020.zarr', 
        f'persistence_{variable}_2020.zarr'
    ]
    
    forecast_times = None
    
    for pattern in forecast_patterns:
        fct_path = os.path.join(input_dir, pattern)
        if os.path.exists(fct_path):
            print(f"\nProcessing forecast: {pattern}")
            forecast = open_fct(fct_path)
            
            # Store forecast times for climatology processing
            if forecast_times is None:
                forecast_times = forecast.time.values
            
            cma_vals = cma_per_lat(forecast, obs, variable)
            model_name = get_model_name_from_path(pattern)
            
            # Save results
            cma_file = os.path.join(output_dir, f'cma_{model_name}.txt')
            
            np.savetxt(cma_file, cma_vals)
            
            print(f"Saved CMA results to: {cma_file}")
        else:
            print(f"Warning: Forecast file not found: {fct_path}")

    # Process climatology
    clim_path = os.path.join(input_dir, f'clim_{variable}_2020.zarr')
    if os.path.exists(clim_path):
        print(f"\nProcessing climatology: {clim_path}")
        climatology = open_fct(clim_path)
        
        if forecast_times is None:
            raise ValueError("No forecast files were processed successfully. Cannot determine forecast times for climatology.")
        
        cma_vals = clim_cma(climatology, obs, forecast_times, variable)
        
        # Save climatology results
        cma_file = os.path.join(output_dir, f'cma_climatology.txt')
        
        np.savetxt(cma_file, cma_vals)
        
        print(f"Saved climatology CMA results to: {cma_file}")
    else:
        print(f"Warning: Climatology file not found: {clim_path}")

    print(f"\nCMA computation complete for variable: {variable}")
    print(f"Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Compute CMA values for weather forecasts')
    parser.add_argument('--input_dir', type=str, default='./fct_data/', 
                       help='Input directory containing the forecast and observation data')
    
    args = parser.parse_args()
    
    # Ensure input directory exists and has proper format
    input_dir = args.input_dir.rstrip('/')
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    print(f"Processing data from: {input_dir}")
    compute_cma(input_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())