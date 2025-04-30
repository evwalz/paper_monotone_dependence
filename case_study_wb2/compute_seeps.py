import xarray as xr
import numpy as np
import pandas as pd
import sys
import argparse
import os


def compute_seeps_clim_per_lat(climatology, obs, climatology_seeps, fct_time_values,  
                             precip_name="total_precipitation_24hr", 
                             dry_threshold_mm=0.25, min_p1=0.1, max_p1=0.85, spatial_avg = True):
    """
    Create a climatological forecast for evaluation.
    
    Args:
        climatology: Climatology dataset with dimensions day_of_year × hours × latitude × longitude
        obs: Observation dataset (used for time alignment)
        fct_time_values: Forecast initialization times
    
    Returns:
        xr.Dataset: Climatological forecast aligned with verification times
    """
    obs_times = obs.time.values
    
    # Calculate the target verification times (t+24h)
    valid_times = []
    valid_target_times = []
    
    for t in fct_time_values:
        if t + np.timedelta64(24, 'h') in obs_times:
            valid_times.append(t)
            valid_target_times.append(t + np.timedelta64(24, 'h'))
    
    # Create arrays for day of year and hour
    doys = np.array([pd.Timestamp(t).dayofyear for t in valid_target_times])
    hours = np.array([pd.Timestamp(t).hour for t in valid_target_times])
    
    # Extract climatology values for each target time
    var = 'total_precipitation_24hr'
    clim_values = []
    
    # Extract climatology values for each verification time
    for i, (doy, hour) in enumerate(zip(doys, hours)):
        # Select the closest hour in the climatology
        hour_idx = np.argmin(np.abs(climatology.hour.values - hour))
        time_slice = climatology[var].sel(dayofyear=doy, hour=climatology.hour.values[hour_idx])
        clim_values.append(time_slice)
    
    # Combine into a single dataset with time dimension
    clim_forecast = xr.concat(clim_values, dim='time')
    clim_forecast = clim_forecast.assign_coords(time=valid_target_times)
    ground_truth = obs.sel(time=valid_target_times)
    # Create a dataset with the same structure as other forecasts
    #climatology_forecast = xr.Dataset({var: clim_forecast})
    dry_threshold = dry_threshold_mm / 1000.0
    
    # Get climatological dry fraction (p1)
    dry_fraction = climatology_seeps[f"{precip_name}_seeps_dry_fraction"]
    p1 = dry_fraction.mean(("hour", "dayofyear")).compute()
    
    # Function to convert precipitation to SEEPS categories
    def convert_precip_to_seeps_cat(da):
        wet_threshold = climatology_seeps[f"{precip_name}_seeps_threshold"]
        
        # Select wet threshold for valid time
        wet_threshold_for_valid_time = wet_threshold.sel(
            dayofyear=da.time.dt.dayofyear,
            hour=da.time.dt.hour
        ).load()
        
        # Categorize precipitation
        dry = da < dry_threshold
        light = np.logical_and(da > dry_threshold, da < wet_threshold_for_valid_time)
        heavy = da >= wet_threshold_for_valid_time
        
        # Combine categories
        result = xr.concat(
            [dry, light, heavy],
            dim=xr.DataArray(["dry", "light", "heavy"], dims=["seeps_cat"]),
        )
        
        # Convert NaNs back to NaNs
        result = result.astype("int").where(da.notnull())
        return result
    
    # Extract forecast and observation variables
    var_forecast = clim_forecast
    var_obs = ground_truth[precip_name]
    
    # Make sure the time dimensions match
    #var_obs = var_obs.sel(time=slice(var_forecast.time.min(), var_forecast.time.max()))
    
    # Handle potentially missing values
    
    #common_times = np.intersect1d(var_forecast.time, var_obs.time)
    #var_forecast = var_forecast.sel(time=common_times)
    #var_obs = var_obs.sel(time=common_times)
    
    # Convert to SEEPS categories
    forecast_cat = convert_precip_to_seeps_cat(var_forecast)
    obs_cat = convert_precip_to_seeps_cat(var_obs)
    
    # Compute contingency table
    print('starting expensive out computation')
    out = (
        forecast_cat.rename({"seeps_cat": "forecast_cat"})
        * obs_cat.rename({"seeps_cat": "truth_cat"})
    ).compute()
    
    # Compute scoring matrix
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
    print('starting expensive for loop')
    for mat in scoring_matrix:
        das.append(xr.concat(mat, dim=out.truth_cat))
    scoring_matrix = 0.5 * xr.concat(das, dim=out.forecast_cat)
    scoring_matrix = scoring_matrix.compute()
    
    # Take dot product
    print('starting expensive dot product')
    result = xr.dot(out, scoring_matrix, dims=("forecast_cat", "truth_cat"))
    
    # Mask out p1 thresholds
    result = result.where(p1 < max_p1, np.nan)
    result = result.where(p1 > min_p1, np.nan)
    
    # Create output dataset
    result_ds = xr.Dataset({f"{precip_name}": result})

    result_ds_lon = result_ds.mean('longitude')
    result_ds_time = result_ds_lon.mean('time')
        
    return result_ds_time 

#########################################################################
#########################################################################
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
    

def compute_seeps_per_lat(forecast, obs, climatology, precip_name="total_precipitation_24hr", 
                 dry_threshold_mm=0.25, min_p1=0.1, max_p1=0.85):
    """
    Compute Stable Equitable Error in Probability Space (SEEPS) metric.
    
    Parameters
    ----------
    forecast : xr.Dataset
        Dataset containing forecast precipitation.
    obs : xr.Dataset
        Dataset containing observed precipitation.
    climatology : xr.Dataset
        Climatology dataset containing seeps_threshold [meters] and
        seeps_dry_fraction [0-1] for given precip_name.
    precip_name : str, optional
        Name of precipitation variable, default is "total_precipitation_24hr".
    dry_threshold_mm : float, optional
        Dry threshold in mm, default is 0.25.
    min_p1 : float, optional
        Mask out values with smaller average dry fraction, default is 0.1.
    max_p1 : float, optional
        Mask out values with larger average dry fraction, default is 0.85.
    spatial_avg : bool, optional
        Whether to perform spatial averaging, default is True.
    region : Region object, optional
        Region to compute SEEPS for, default is None.
        
    Returns
    -------
    xr.Dataset
        Dataset containing SEEPS score.
    """
    # Convert dry threshold to meters (SI units)
    dry_threshold = dry_threshold_mm / 1000.0
    
    # Get climatological dry fraction (p1)
    dry_fraction = climatology[f"{precip_name}_seeps_dry_fraction"]
    #p1 = dry_fraction.mean(("hour", "dayofyear")).compute()
    p1 = dry_fraction.mean(("hour", "dayofyear")).compute()
    
    # Convert precipitation to SEEPS categories (dry, light, heavy)
    def convert_precip_to_seeps_cat(da):
        wet_threshold = climatology[f"{precip_name}_seeps_threshold"]
        #da = ds[precip_name]
        
        # Select wet threshold for valid time
        wet_threshold_for_valid_time = wet_threshold.sel(
            dayofyear=da.time.dt.dayofyear, 
            hour=da.time.dt.hour
        ).load()
        
        # Categorize precipitation
        dry = da < dry_threshold
        light = np.logical_and(da > dry_threshold, da < wet_threshold_for_valid_time)
        heavy = da >= wet_threshold_for_valid_time
        
        # Combine categories
        result = xr.concat(
            [dry, light, heavy],
            dim=xr.DataArray(["dry", "light", "heavy"], dims=["seeps_cat"]),
        )
        
        # Convert NaNs back to NaNs
        result = result.astype("int").where(da.notnull())
        return result
    
    # Convert forecast and observations to categories
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

    forecast_cat = convert_precip_to_seeps_cat(var_forecast)
    obs_cat = convert_precip_to_seeps_cat(var_obs)
    
    # Compute contingency table
    print('starting expensive out computation')
    out = (
        forecast_cat.rename({"seeps_cat": "forecast_cat"})
        * obs_cat.rename({"seeps_cat": "truth_cat"})
    ).compute()
    
    # Compute scoring matrix
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
    print('starting expensive for loop')
    for mat in scoring_matrix:
        das.append(xr.concat(mat, dim=out.truth_cat))
    scoring_matrix = 0.5 * xr.concat(das, dim=out.forecast_cat)
    scoring_matrix = scoring_matrix.compute()
    
    # Take dot product
    print('starting expensive dot product')
    result = xr.dot(out, scoring_matrix, dims=("forecast_cat", "truth_cat"))
    
    # Mask out p1 thresholds
    result = result.where(p1 < max_p1, np.nan)
    result = result.where(p1 > min_p1, np.nan)
    
    # Create output dataset
    result_ds = xr.Dataset({f"{precip_name}": result})
    
    result_ds_lon = result_ds.mean('longitude')
    result_ds_time = result_ds_lon.mean('time')
        
    return result_ds_time

def compute_seeps(input_dir):
    output_dir = input_dir + '/seeps_results/'
    os.makedirs(output_dir, exist_ok = True)
    
    obs_path = input_dir + 'era5_obs_precipitation_24hr_2020.zarr'
    obs = open_obs(obs_path)

    #climatology = xr.open_dataset('./precip_data/global_seeps_climatology.nc')
    climatology = xr.open_zarr("gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr")#xr.open_dataset('./precip_data/global_seeps_climatology.nc')
    climatology = make_latitude_increasing(climatology)

    forecast_paths = ['graphcast_ifs_precipitation_24hr_2020.zarr', 'ifs_hres_precipitation_24hr_2020.zarr', 'ifs_mean_precipitation_24hr_2020.zarr', 'persistence_precipitation_24hr_2020.zarr']

    seeps_vals_list = list()
    for fct_path in forecast_paths:
        forecast = open_fct(input_dir + fct_path)
        seeps_lat = compute_seeps_per_lat(forecast, obs, climatology)
        save_name = get_letters_until_second_underscore(fct_path)
        seeps_lat.to_netcdf(output_dir + 'seeps_'+save_name+'.nc')

    time_fct = forecast.time.values

    # Climatology:
    climatology_forecast = open_fct(input_dir + 'clim_precipitation_24hr_2020.zarr')
    seeps_lat = compute_seeps_clim_per_lat(climatology_forecast, obs, climatology, time_fct)
    seeps_lat.to_netcdf(output_dir + 'seeps_climatology.nc')


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_dir', type=str, default='./fct_data/', help='Output file path')
    args = parser.parse_args()
    input_dir = args.input_dir
    compute_seeps(input_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())