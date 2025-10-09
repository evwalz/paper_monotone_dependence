import xarray as xr
import numpy as np
import os
import pandas as pd
import sys
import argparse
import time
import glob

def convert_precip_to_seeps_cat(da, climatology, variable, dry_threshold):
    """
    Convert precipitation to SEEPS categories (dry, light, heavy).
    
    Parameters
    ----------
    da : xr.DataArray
        Precipitation data array with time dimension
    climatology : xr.Dataset
        Climatology dataset containing SEEPS thresholds
    variable : str
        Variable name
    dry_threshold : float
        Dry threshold in meters (SI units)
    
    Returns
    -------
    xr.DataArray
        Categorical array with dimension seeps_cat=['dry', 'light', 'heavy']
    """
    wet_threshold = climatology[f"{variable}_seeps_threshold"]
    
    # Select wet threshold for valid time
    wet_threshold_for_valid_time = wet_threshold.sel(
        dayofyear=da.time.dt.dayofyear,
        hour=da.time.dt.hour
    ).load()
    
    # Categorize precipitation
    dry = da < dry_threshold
    light = np.logical_and(da >= dry_threshold, da < wet_threshold_for_valid_time)
    heavy = da >= wet_threshold_for_valid_time
    
    # Combine categories
    result = xr.concat(
        [dry, light, heavy],
        dim=xr.DataArray(["dry", "light", "heavy"], dims=["seeps_cat"]),
    )
    
    # Convert NaNs back to NaNs
    result = result.astype("int").where(da.notnull())
    return result

def _compute_seeps_grid(var_forecast, var_obs, climatology, variable, dry_threshold, 
                        min_p1, max_p1, start_time):
    """Helper function to compute SEEPS per latitude."""
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
    
    # Get climatological dry fraction (p1)
    dry_fraction = climatology[f"{variable}_seeps_dry_fraction"]
    p1 = dry_fraction.mean(("hour", "dayofyear")).compute()
    
    # Convert to SEEPS categories
    #print("Converting forecast to SEEPS categories...")
    forecast_cat = convert_precip_to_seeps_cat(var_forecast, climatology, variable, dry_threshold)
    
    #print("Converting observations to SEEPS categories...")
    obs_cat = convert_precip_to_seeps_cat(var_obs, climatology, variable, dry_threshold)
    
    # Compute contingency table
    #print("Computing contingency table...")
    out = (
        forecast_cat.rename({"seeps_cat": "forecast_cat"})
        * obs_cat.rename({"seeps_cat": "truth_cat"})
    ).compute()
    
    # Compute scoring matrix
    #print("Computing scoring matrix...")
    scoring_matrix = [
        [xr.zeros_like(p1), 1 / (1 - p1), 4 / (1 - p1)],
        [1 / p1, xr.zeros_like(p1), 3 / (1 - p1)],
        [
            1 / p1 + 3 / (2 + p1),
            3 / (2 + p1),
            xr.zeros_like(p1),
        ],
    ]
    
    das = []
    for mat in scoring_matrix:
        das.append(xr.concat(mat, dim=out.truth_cat))
    scoring_matrix = 0.5 * xr.concat(das, dim=out.forecast_cat)
    scoring_matrix = scoring_matrix.compute()
    
    # Take dot product
    #print("Computing SEEPS scores...")
    result = xr.dot(out, scoring_matrix, dims=("forecast_cat", "truth_cat"))
    
    # Mask out p1 thresholds
    result = result.where(p1 < max_p1, np.nan)
    result = result.where(p1 > min_p1, np.nan)
    
    # Average over longitude and time
    result_per_lat = result.mean(dim=[lon_name, 'time'])
    
    total_time = time.time() - start_time
    print(f"SEEPS computation complete in {total_time:.2f} seconds.")
    
    return result_per_lat.values

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

def seeps_per_lat(forecast, obs, climatology_seeps, variable, lead_time_hours,
                  dry_threshold_mm=0.25, min_p1=0.1, max_p1=0.85):
    """
    Compute SEEPS per latitude using the reference approach:
    1. Shift forecast time coordinates by lead_time (init time -> valid time)
    2. Align forecast and observations on valid time
    3. Compute SEEPS
    
    Parameters
    ----------
    forecast : xr.Dataset
        Forecast dataset
    obs : xr.Dataset
        Observation dataset
    climatology_seeps : xr.Dataset
        SEEPS climatology with thresholds and dry fraction
    variable : str
        Variable name (must be total_precipitation_24hr)
    lead_time_hours : int
        Lead time in hours
    dry_threshold_mm : float
        Dry threshold in mm (default: 0.25)
    min_p1 : float
        Minimum dry fraction threshold (default: 0.1)
    max_p1 : float
        Maximum dry fraction threshold (default: 0.85)
    
    Returns
    -------
    np.ndarray
        SEEPS values per latitude
    """
    start_time = time.time()
    
    # Convert dry threshold to meters (SI units)
    dry_threshold = dry_threshold_mm / 1000.0
    
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

    
    return _compute_seeps_grid(var_forecast, var_obs, climatology_seeps, variable, 
                               dry_threshold, min_p1, max_p1, start_time)

def clim_seeps(climatology_forecast, obs, climatology_seeps, forecast_init_times, 
               variable, lead_time_hours, dry_threshold_mm=0.25, min_p1=0.1, max_p1=0.85):
    """
    Compute SEEPS for climatology using the reference approach.
    """
    start_time = time.time()
    
    # Convert dry threshold to meters
    dry_threshold = dry_threshold_mm / 1000.0
    
    # Calculate valid times from forecast initialization times
    lead_time_delta = np.timedelta64(lead_time_hours, 'h')
    valid_times = forecast_init_times + lead_time_delta
    
    # Filter to times that exist in observations
    valid_times = [t for t in valid_times if t in obs.time.values]
    
    # Get observations at valid times
    obs_valid = obs.sel(time=valid_times)
    
    # Create a time selection dictionary for climatology forecast
    time_selection = {
        'dayofyear': xr.DataArray([pd.Timestamp(t).dayofyear for t in valid_times], dims='time')
    }
    
    # Add hour selection if climatology includes hourly data
    if 'hour' in climatology_forecast.coords:
        time_selection['hour'] = xr.DataArray([pd.Timestamp(t).hour for t in valid_times], dims='time')

    # Select climatology forecast values based on the time selection
    clim_forecast = climatology_forecast[variable].sel(time_selection)
    
    # Assign the valid times as coordinates to climatology
    clim_forecast = clim_forecast.assign_coords(time=valid_times)
    
    # Get the observations for the same variable
    var_obs = obs_valid[variable]
    
    # Make sure time dimensions match
    common_times = np.intersect1d(clim_forecast.time, var_obs.time)
    var_forecast = clim_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)


    return _compute_seeps_grid(var_forecast, var_obs, climatology_seeps, variable,
                               dry_threshold, min_p1, max_p1, start_time)

def compute_seeps(input_dir, lead_times=[24, 48, 72], dry_threshold_mm=0.25, 
                  min_p1=0.1, max_p1=0.85):
    """
    Compute SEEPS for precipitation forecasts at multiple lead times.
    
    Parameters
    ----------
    input_dir : str
        Input directory containing forecast and observation data
    lead_times : list
        Lead times in hours to compute SEEPS for
    dry_threshold_mm : float
        Dry threshold in mm
    min_p1 : float
        Minimum dry fraction threshold
    max_p1 : float
        Maximum dry fraction threshold
    """
    output_dir = os.path.join(input_dir, 'seeps_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # SEEPS is only for precipitation
    variable = 'total_precipitation_24hr'
    
    obs_path = os.path.join(input_dir, f'era5_obs_{variable}_2020.zarr')
    
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Observation file not found: {obs_path}")
    
    #print(f"Loading observations from: {obs_path}")
    obs = open_obs(obs_path)
    
    # Load SEEPS climatology (this is always the same, independent of lead time)
    #print("\nLoading SEEPS climatology...")
    climatology_seeps_path = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
    climatology_seeps = xr.open_zarr(climatology_seeps_path)
    climatology_seeps = make_latitude_increasing(climatology_seeps)
    #print("SEEPS climatology loaded successfully")
    
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
                
                seeps_vals = seeps_per_lat(forecast, obs, climatology_seeps, variable, 
                                          lead_time_hours, dry_threshold_mm, min_p1, max_p1)
                model_name = get_model_name_from_path(pattern)
                
                # Save results
                seeps_file = os.path.join(output_dir, f'seeps_{model_name}_{variable}_lead{lead_time_hours}h.txt')
                
                np.savetxt(seeps_file, seeps_vals)
                
                print(f"Saved SEEPS results to: {seeps_file}")
            else:
                print(f"Warning: Forecast file not found: {fct_path}")

        # Process climatology for this lead time
        clim_path = os.path.join(input_dir, f'clim_{variable}_{lead_time_hours}h_2020.zarr')
        if os.path.exists(clim_path):
            print(f"\nProcessing climatology: {clim_path} at {lead_time_hours}h lead time")
            climatology_forecast = open_fct(clim_path)
            
            if forecast_init_times is None:
                raise ValueError("No forecast files were processed successfully. Cannot determine forecast times for climatology.")
            
            seeps_vals = clim_seeps(climatology_forecast, obs, climatology_seeps, 
                                   forecast_init_times, variable, lead_time_hours,
                                   dry_threshold_mm, min_p1, max_p1)
            
            # Save climatology results
            seeps_file = os.path.join(output_dir, f'seeps_climatology_{variable}_lead{lead_time_hours}h.txt')
            
            np.savetxt(seeps_file, seeps_vals)
            
            print(f"Saved climatology SEEPS results to: {seeps_file}")
        else:
            print(f"Warning: Climatology file not found: {clim_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute SEEPS (Stable Equitable Error in Probability Space) for precipitation forecasts at multiple lead times'
    )
    parser.add_argument('--input_dir', type=str, 
                       default='./fct_data/', 
                       help='Input directory containing the forecast and observation data')
    parser.add_argument('--lead_times', type=int, nargs='+', default=[24],
                       help='Lead times in hours to compute SEEPS for (default: 24 48 72)')
    parser.add_argument('--dry_threshold_mm', type=float, default=0.25,
                       help='Dry threshold in mm (default: 0.25)')
    parser.add_argument('--min_p1', type=float, default=0.1,
                       help='Minimum dry fraction threshold (default: 0.1)')
    parser.add_argument('--max_p1', type=float, default=0.85,
                       help='Maximum dry fraction threshold (default: 0.85)')
    
    args = parser.parse_args()
    
    # Ensure input directory exists and has proper format
    input_dir = args.input_dir.rstrip('/')
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    compute_seeps(input_dir, args.lead_times, args.dry_threshold_mm, 
                 args.min_p1, args.max_p1)
    return 0

if __name__ == "__main__":
    sys.exit(main())
