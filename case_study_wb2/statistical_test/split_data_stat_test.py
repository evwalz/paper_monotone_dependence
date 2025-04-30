# split_grid.py - Script to split the grid into chunks
import numpy as np
import xarray as xr
import os
import pickle
from pathlib import Path
from statistical_test.config import FORECAST_NAMES, DATA_DIR, CHUNK_DIR, NUM_CHUNKS, OBS_PATH

def standardize_dims(dataset: xr.Dataset) -> xr.Dataset:
    """Standardize dimension names to time, latitude, longitude."""
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
    
    if dim_mapping:
        return dataset.rename(dim_mapping)
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

def open_zarr_fct_all(forecast_path: str, obs_path: str, forecast_names):
    """Open and standardize datasets from zarr files."""
    obs = xr.open_zarr(obs_path)
    obs = standardize_dims(obs)
    obs = make_latitude_increasing(obs)
    
    def load_fct(name, obs):
        forecast = xr.open_zarr(forecast_path + name + '_precipitation_24hr_2020.zarr')
        forecast = standardize_dims(forecast)
        forecast = make_latitude_increasing(forecast)

        # Ensure the grids are aligned (you may need to implement this function)
        # forecast = ensure_aligned_grid(forecast, obs)
        return forecast
    
    forecast1 = load_fct(forecast_names[0], obs)
    forecast2 = load_fct(forecast_names[1], obs)
    #forecast3 = load_fct(forecast_names[2], obs)
    return forecast1, forecast2, obs #forecast3, obs

def prepare_data_chunks(forecast_path, obs_path, forecast_names, n_chunks, chunk_dir='./chunks'):
    """
    Prepare data chunks for distributed processing.
    
    Parameters:
    -----------
    forecast_path : str
        Path to forecast data
    obs_path : str
        Path to observation data
    forecast_names : list
        List of forecast names
    n_chunks : int
        Number of chunks to create
    chunk_dir : str
        Directory to save chunks
        
    Returns:
    --------
    tuple
        Grid dimensions and chunk information
    """
    # Create chunk directory if it doesn't exist
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Load data
    forecast_graphcast, forecast_ifs_hres, obs = open_zarr_fct_all(
        forecast_path, obs_path, forecast_names
    )
    
    # Calculate observation times that match forecast times
    forecast_times = forecast_graphcast.time.values
    obs_times = obs.time.values
    
    valid_forecast_times = [t for t in forecast_times if t + np.timedelta64(24, 'h') in obs_times]
    valid_obs_times = [t + np.timedelta64(24, 'h') for t in valid_forecast_times]
    
    filtered_obs = obs.sel(time=valid_obs_times)
    latitudes = obs.latitude.values
    longitudes = obs.longitude.values
    
    # Prepare observation data
    var = 'total_precipitation_24hr'
    var_obs = filtered_obs[var].values
    
    # Prepare forecast data
    def modify_fct(forecast, valid_forecast_times, valid_obs_times):
        filtered_forecast = forecast.sel(time=valid_forecast_times)
        filtered_forecast = filtered_forecast.assign_coords(time=valid_obs_times)
        var_forecast = filtered_forecast['total_precipitation_24hr'].values
        return var_forecast
    
    fct1 = modify_fct(forecast_graphcast, valid_forecast_times, valid_obs_times)
    fct2 = modify_fct(forecast_ifs_hres, valid_forecast_times, valid_obs_times)
    
    # Calculate chunk sizes
    n_lat = len(latitudes)
    n_lon = len(longitudes)
    
    total_points = n_lat * n_lon
    base_size = total_points // n_chunks
    remainder = total_points % n_chunks
    chunks = []
    start_idx = 0
    for i in range(n_chunks):
        # Add one extra point to chunks until remainder is used up
        chunk_size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        chunks.append((start_idx, end_idx))
        start_idx = end_idx
    
    # Save grid dimensions and chunk info
    grid_info = {
        'n_lat': n_lat,
        'n_lon': n_lon,
        'latitudes': latitudes,
        'longitudes': longitudes,
        'chunks': chunks,
        'forecast_names': forecast_names  # Save forecast names in grid_info
    }
    
    with open(f'{chunk_dir}/grid_info.pkl', 'wb') as f:
        pickle.dump(grid_info, f)
    
    # Save data arrays
    np.save(f'{chunk_dir}/fct1.npy', fct1)
    np.save(f'{chunk_dir}/fct2.npy', fct2)
    np.save(f'{chunk_dir}/var_obs.npy', var_obs)
    
    print(f"Data split into {len(chunks)} chunks and saved to {chunk_dir}")
    print(f"Total grid points: {total_points}, Points covered: {chunks[-1][1]}")
    return n_lat, n_lon, chunks

if __name__ == "__main__":
    # Configure paths and settings - now using the central config
    print(f"Using forecast names: {FORECAST_NAMES}")
    
    # Prepare data chunks
    prepare_data_chunks(DATA_DIR, OBS_PATH, FORECAST_NAMES, NUM_CHUNKS, CHUNK_DIR)