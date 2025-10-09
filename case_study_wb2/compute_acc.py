import xarray as xr
import numpy as np
import os
import pandas as pd
import sys
import argparse
import time
import glob

def _compute_acc_grid(var_forecast, var_obs, clim_for_times, start_time):
    """Helper function to compute ACC per latitude."""
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
    
    #print(f"Using spatial coordinates: {lat_name}, {lon_name}")
    
    latitudes = var_forecast[lat_name].values

    #print(f"Data prepared in {time.time() - start_time:.2f} seconds")
    #print(f"Computing ACC for {len(latitudes)} latitudes...")
    
    # Compute anomalies
    forecast_anom = var_forecast - clim_for_times
    truth_anom = var_obs - clim_for_times
    
    # Compute ACC using xarray operations
    # ACC = mean(forecast_anom * truth_anom) / (std(forecast_anom) * std(truth_anom))
    numerator = (forecast_anom * truth_anom).mean(lon_name, skipna=True)
    forecast_std = np.sqrt((forecast_anom ** 2).mean(lon_name, skipna=True))
    truth_std = np.sqrt((truth_anom ** 2).mean(lon_name, skipna=True))
    
    # Compute ACC
    acc = numerator / (forecast_std * truth_std)
    
    # Average over time to get ACC per latitude
    acc_per_lat = acc.mean(dim='time')
    
    total_time = time.time() - start_time
    print(f"ACC computation complete in {total_time:.2f} seconds.")
    
    return acc_per_lat.values

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
    #print(f"  Original dims: {list(obs.dims.keys())}")
    #print(f"  Original coords: {list(obs.coords.keys())}")
    obs = standardize_dims(obs)
    #print(f"  After standardize dims: {list(obs.dims.keys())}")
    #print(f"  After standardize coords: {list(obs.coords.keys())}")
    obs = make_latitude_increasing(obs)
    #print(f"  After make_latitude_increasing dims: {list(obs.dims.keys())}")
    return obs

def open_fct(forecast_path):
    forecast = xr.open_zarr(forecast_path)
    #print(f"  Original dims: {list(forecast.dims.keys())}")
    #print(f"  Original coords: {list(forecast.coords.keys())}")
    forecast = standardize_dims(forecast)
    #print(f"  After standardize dims: {list(forecast.dims.keys())}")
    #print(f"  After standardize coords: {list(forecast.coords.keys())}")
    forecast = make_latitude_increasing(forecast)
    #print(f"  After make_latitude_increasing dims: {list(forecast.dims.keys())}")
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

def acc_per_lat(forecast, obs, climatology, variable, lead_time_hours):
    """
    Compute ACC (Anomaly Correlation Coefficient) per latitude using the reference approach:
    1. Shift forecast time coordinates by lead_time (init time -> valid time)
    2. Align forecast and observations on valid time
    3. Get climatology for valid times
    4. Compute anomalies and ACC
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
    
    #print(f"Variable: {variable}")
    #print(f"Lead time: {lead_time_hours} hours")
    #print(f"Number of verification times: {len(common_times)}")
    #print(f"First verification time: {common_times[0]}")
    #print(f"Last verification time: {common_times[-1]}")
    
    # Step 4: Get climatology for valid times
    clim_var = climatology[variable]
    
    # Select climatology based on day of year and hour of valid times
    time_selection = dict(
        dayofyear=var_forecast.time.dt.dayofyear
    )
    
    if "hour" in climatology.coords:
        time_selection["hour"] = var_forecast.time.dt.hour
    
    clim_for_times = clim_var.sel(time_selection).load()
    
    return _compute_acc_grid(var_forecast, var_obs, clim_for_times, start_time)

def clim_acc(climatology, obs, forecast_init_times, variable, lead_time_hours):
    """
    Compute ACC for climatology using the reference approach.
    
    For climatology, ACC is always 0 by definition because:
    - Climatology has no temporal anomaly correlation with observations
    - The climatology is the reference state, so climatology - climatology = 0
    """
    start_time = time.time()
    
    # Calculate valid times from forecast initialization times
    lead_time_delta = np.timedelta64(lead_time_hours, 'h')
    valid_times = forecast_init_times + lead_time_delta
    
    # Filter to times that exist in observations
    valid_times = [t for t in valid_times if t in obs.time.values]
    
    #print(f"Variable: {variable}")
    #print(f"Lead time: {lead_time_hours} hours")
    #print(f"Number of verification times: {len(valid_times)}")
    
    # Get the number of latitudes from observations
    lat_name = next((dim for dim in ['latitude', 'lat'] if dim in obs.dims), None)
    num_lats = len(obs[lat_name])
    
    # ACC for climatology is 0 by definition
    acc_vals = np.zeros(num_lats, dtype=np.float64)
    
    #total_time = time.time() - start_time
    #print(f"ACC computation for climatology complete in {total_time:.2f} seconds.")
    #print(f"ACC for climatology is 0 by definition (returning zeros)")
    
    return acc_vals

def compute_acc(input_dir, variable=None, lead_times=[24, 48, 72]):
    """Compute ACC for all forecast models for a specific variable at multiple lead times."""
    output_dir = os.path.join(input_dir, 'acc_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # If variable not specified, try to auto-detect from available files
    if variable is None:
        obs_files = glob.glob(os.path.join(input_dir, 'era5_obs_*.zarr'))
        if not obs_files:
            raise ValueError("No ERA5 observation files found. Please specify variable explicitly.")
        
        obs_path = obs_files[0]
        variable = get_variable_from_path(obs_path)
        #print(f"Auto-detected variable: {variable}")
    else:
        obs_path = os.path.join(input_dir, f'era5_obs_{variable}_2020.zarr')
    
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Observation file not found: {obs_path}")
    
    #print(f"Loading observations from: {obs_path}")
    obs = open_obs(obs_path)
    
    # Loop over each lead time
    for lead_time_hours in lead_times:
        #print(f"\n{'='*80}")
        #print(f"COMPUTING ACC FOR LEAD TIME: {lead_time_hours} HOURS")
        #print(f"{'='*80}\n")
        
        # Load climatology for this lead time
        clim_path = os.path.join(input_dir, f'clim_{variable}_{lead_time_hours}h_2020.zarr')
        if not os.path.exists(clim_path):
            print(f"Warning: Climatology file not found: {clim_path}")
            print("Climatology is required for ACC computation. Skipping this lead time.")
            continue
        
        print(f"Loading climatology from: {clim_path}")
        climatology = open_fct(clim_path)
        
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
                
                acc_vals = acc_per_lat(forecast, obs, climatology, variable, lead_time_hours)
                model_name = get_model_name_from_path(pattern)
                
                # Save results
                acc_file = os.path.join(output_dir, f'acc_{model_name}_{variable}_lead{lead_time_hours}h.txt')
                
                np.savetxt(acc_file, acc_vals)
                
                print(f"Saved ACC results to: {acc_file}")
            else:
                print(f"Warning: Forecast file not found: {fct_path}")

        # Process climatology ACC (always 0)
        if forecast_init_times is not None:
            #print(f"\nProcessing climatology ACC at {lead_time_hours}h lead time")
            
            acc_vals = clim_acc(climatology, obs, forecast_init_times, variable, lead_time_hours)
            
            # Save climatology results
            acc_file = os.path.join(output_dir, f'acc_climatology_{variable}_lead{lead_time_hours}h.txt')
            
            np.savetxt(acc_file, acc_vals)
            
            print(f"Saved climatology ACC results to: {acc_file}")
        else:
            print(f"Warning: No forecast files were processed successfully. Cannot compute climatology ACC.")

    #print(f"\n{'='*80}")
    #print(f"ACC COMPUTATION COMPLETE FOR VARIABLE: {variable}")
    #print(f"Lead times processed: {lead_times}")
    #print(f"Results saved to: {output_dir}")
    #print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Compute ACC (Anomaly Correlation Coefficient) values for weather forecasts at multiple lead times')
    parser.add_argument('--input_dir', type=str, default='./fct_data/', 
                       help='Input directory containing the forecast and observation data')
    parser.add_argument('--variable', type=str, default='total_precipitation_24hr',
                       help='Variable to process. Options: 2m_temperature, 10m_wind_speed, mean_sea_level_pressure, total_precipitation_24hr.')
    parser.add_argument('--lead_times', type=int, nargs='+', default=[24],
                       help='Lead times in hours to compute ACC for')
    
    args = parser.parse_args()
    
    # Ensure input directory exists and has proper format
    input_dir = args.input_dir.rstrip('/')
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    #print(f"Processing data from: {input_dir}")
    #print(f"Lead times: {args.lead_times}")
    compute_acc(input_dir, args.variable, args.lead_times)
    return 0

if __name__ == "__main__":
    sys.exit(main())
