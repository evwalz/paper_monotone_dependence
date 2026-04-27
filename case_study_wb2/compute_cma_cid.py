"""
Compute CMA and/or CID (C-index) from zarr forecasts vs ERA5 observations.

Uses the ``acor`` package: ``acor(forecast, obs, method="cma")`` and
``acor(forecast, obs, method="cid")`` (x = predictor, y = outcome).

Entry point: ``python compute_cma_cid.py`` with ``--metric cma|cindx|both``.
"""

import xarray as xr
import numpy as np
import os
import pandas as pd
import sys
import argparse
import time
import glob

try:
    from acor import acor
except ImportError as e:
    raise ImportError(
        "compute_cma_cid.py requires the 'acor' package. Install e.g.:\n"
        '  pip install "git+https://github.com/evwalz/acor-python.git"\n'
        "See case_study_wb2/statistical_test/README.md and acor-python docs."
    ) from e


def _acor_point_1d(fct: np.ndarray, obs: np.ndarray, method: str) -> float:
    """CMA or CID for one grid cell; x = forecast, y = obs (acor API)."""
    fct = np.asarray(fct, dtype=np.float64).ravel()
    obs = np.asarray(obs, dtype=np.float64).ravel()
    if fct.shape != obs.shape:
        return np.nan
    valid = np.isfinite(fct) & np.isfinite(obs)
    if int(np.sum(valid)) < 2:
        return np.nan
    fct_v, obs_v = fct[valid], obs[valid]
    if len(np.unique(obs_v)) < 2:
        return np.nan
    try:
        res = acor(fct_v, obs_v, method=method)
        return float(np.asarray(res.statistic).ravel()[0])
    except (ValueError, Exception):
        return np.nan


def _compute_acor_metric_grid(var_forecast, var_obs, start_time, method: str):
    """CMA (method='cma') or CID (method='cid') on a lat-lon x time field."""
    if method not in ("cma", "cid"):
        raise ValueError("method must be 'cma' or 'cid'")
    label = "CMA" if method == "cma" else "CID (cindx)"

    lat_name = None
    lon_name = None
    for candidate in ("latitude", "lat"):
        if candidate in var_forecast.dims:
            lat_name = candidate
            break
    for candidate in ("longitude", "lon"):
        if candidate in var_forecast.dims:
            lon_name = candidate
            break
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Spatial dimensions not found. Dims: {list(var_forecast.dims)}"
        )

    latitudes = var_forecast[lat_name].values
    longitudes = var_forecast[lon_name].values
    out_grid = np.full((len(latitudes), len(longitudes)), np.nan, dtype=float)
    batch_size = max(1, len(latitudes) // 40)

    for batch_start in range(0, len(latitudes), batch_size):
        batch_end = min(batch_start + batch_size, len(latitudes))
        lat_batch = latitudes[batch_start:batch_end]
        batch_forecast = var_forecast.sel({lat_name: lat_batch})
        batch_obs = var_obs.sel({lat_name: lat_batch})
        for i, lat in enumerate(lat_batch):
            lat_idx = batch_start + i
            fct_data = batch_forecast.sel({lat_name: lat}).values
            obs_data = batch_obs.sel({lat_name: lat}).values
            for j, _ in enumerate(longitudes):
                fct_series = fct_data[:, j]
                obs_series = obs_data[:, j]
                out_grid[lat_idx, j] = _acor_point_1d(
                    fct_series, obs_series, method=method
                )

    per_lat = np.nanmean(out_grid, axis=1)
    print(f"{label} computation complete in {time.time() - start_time:.2f} seconds.")
    return per_lat

def standardize_dims(dataset: xr.Dataset) -> xr.Dataset:
    """Standardize dimension and coordinate names to time, latitude, longitude."""
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
    
    # Also check coordinates (not just dimensions)
    coord_mapping = {}
    if 'latitude' not in dataset.coords:
        if 'lat' in dataset.coords:
            coord_mapping['lat'] = 'latitude'
    
    if 'longitude' not in dataset.coords:
        if 'lon' in dataset.coords:
            coord_mapping['lon'] = 'longitude'
    
    # Apply dimension renaming
    if dim_mapping:
        dataset = dataset.rename(dim_mapping)
    
    # Apply coordinate renaming if different from dimension mapping
    if coord_mapping and coord_mapping != dim_mapping:
        dataset = dataset.rename(coord_mapping)
    
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
    # Expected format: {model}_{variable}_{leadtime}_{year}.zarr
    parts = name_without_ext.split('_')
    
    # Remove known prefixes
    model_prefixes = ['graphcast', 'ifs', 'hres', 'era5', 'obs', 'clim', 'persistence', 'pangu']
    year_suffix = '2020'
    
    # Find the variable part(s) between model and year
    start_idx = 0
    end_idx = len(parts)
    
    # Skip model prefixes
    for i, part in enumerate(parts):
        if part not in model_prefixes:
            start_idx = i
            break
    
    # Find year suffix or lead time pattern (e.g., "24h", "48h")
    for i in range(len(parts)-1, -1, -1):
        if parts[i] == year_suffix or parts[i].endswith('h'):
            end_idx = i
            break
    
    if start_idx < end_idx:
        variable_parts = parts[start_idx:end_idx]
        return '_'.join(variable_parts)
    
    return 'unknown_variable'

def cma_per_lat(forecast, obs, variable, lead_time_hours):
    """
    Compute CMA per latitude using the reference approach:
    1. Shift forecast time coordinates by lead_time (init time -> valid time)
    2. Align forecast and observations on valid time
    3. Compute CMA
    """
    start_time = time.time()
    
    # Step 1: Shift forecast time coordinates to valid time (reference approach)
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
    
    return _compute_acor_metric_grid(var_forecast, var_obs, start_time, "cma")

def clim_cma(climatology, obs, forecast_init_times, variable, lead_time_hours):
    """
    Compute CMA for climatology using the reference approach.
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
    
    #print(f"Variable: {variable}")
    #print(f"Lead time: {lead_time_hours} hours")
    #print(f"Number of verification times: {len(common_times)}")

    return _compute_acor_metric_grid(var_forecast, var_obs, start_time, "cma")


def cindx_per_lat(forecast, obs, variable, lead_time_hours):
    start_time = time.time()
    lead_time_delta = np.timedelta64(lead_time_hours, "h")
    forecast_valid = forecast.assign_coords(time=forecast.time + lead_time_delta)
    var_forecast = forecast_valid[variable]
    var_obs = obs[variable]
    common_times = np.intersect1d(var_forecast.time, var_obs.time)
    var_forecast = var_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)
    return _compute_acor_metric_grid(var_forecast, var_obs, start_time, "cid")


def clim_cindx(climatology, obs, forecast_init_times, variable, lead_time_hours):
    start_time = time.time()
    lead_time_delta = np.timedelta64(lead_time_hours, "h")
    valid_times = forecast_init_times + lead_time_delta
    valid_times = [t for t in valid_times if t in obs.time.values]
    obs_valid = obs.sel(time=valid_times)
    time_selection = {
        "dayofyear": xr.DataArray(
            [pd.Timestamp(t).dayofyear for t in valid_times], dims="time"
        )
    }
    if "hour" in climatology.coords:
        time_selection["hour"] = xr.DataArray(
            [pd.Timestamp(t).hour for t in valid_times], dims="time"
        )
    clim_forecast = climatology[variable].sel(time_selection)
    clim_forecast = clim_forecast.assign_coords(time=valid_times)
    var_obs = obs_valid[variable]
    common_times = np.intersect1d(clim_forecast.time, var_obs.time)
    var_forecast = clim_forecast.sel(time=common_times)
    var_obs = var_obs.sel(time=common_times)
    return _compute_acor_metric_grid(var_forecast, var_obs, start_time, "cid")


def compute_cindx(input_dir, variable=None, lead_times=(24, 48, 72)):
    """Write per-latitude CID series to ``cindx_results/`` (same layout as CMA)."""
    output_dir = os.path.join(input_dir, "cindx_results")
    os.makedirs(output_dir, exist_ok=True)

    if variable is None:
        obs_files = glob.glob(os.path.join(input_dir, "era5_obs_*.zarr"))
        if not obs_files:
            raise ValueError("No ERA5 observation files found.")
        obs_path = obs_files[0]
        variable = get_variable_from_path(obs_path)
    else:
        obs_path = os.path.join(input_dir, f"era5_obs_{variable}_2020.zarr")

    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    obs = open_obs(obs_path)

    for lead_time_hours in lead_times:
        forecast_patterns = [
            f"graphcast_ifs_{variable}_{lead_time_hours}h_2020.zarr",
            f"ifs_hres_{variable}_{lead_time_hours}h_2020.zarr",
            f"persistence_{variable}_{lead_time_hours}h_2020.zarr",
        ]
        forecast_init_times = None

        for pattern in forecast_patterns:
            fct_path = os.path.join(input_dir, pattern)
            if os.path.exists(fct_path):
                forecast = open_fct(fct_path)
                if forecast_init_times is None:
                    forecast_init_times = forecast.time.values
                cindx_vals = cindx_per_lat(forecast, obs, variable, lead_time_hours)
                model_name = get_model_name_from_path(pattern)
                out_file = os.path.join(
                    output_dir,
                    f"cindx_{model_name}_{variable}_lead{lead_time_hours}h.txt",
                )
                np.savetxt(out_file, cindx_vals)
                print(f"Saved CID results to: {out_file}")
            else:
                print(f"Warning: Forecast file not found: {fct_path}")

        clim_path = os.path.join(
            input_dir, f"clim_{variable}_{lead_time_hours}h_2020.zarr"
        )
        if os.path.exists(clim_path):
            climatology = open_fct(clim_path)
            if forecast_init_times is None:
                raise ValueError(
                    "No forecast files processed; cannot align climatology."
                )
            cindx_vals = clim_cindx(
                climatology, obs, forecast_init_times, variable, lead_time_hours
            )
            out_file = os.path.join(
                output_dir,
                f"cindx_climatology_{variable}_lead{lead_time_hours}h.txt",
            )
            np.savetxt(out_file, cindx_vals)
            print(f"Saved climatology CID results to: {out_file}")
        else:
            print(f"Warning: Climatology file not found: {clim_path}")


def compute_cma(input_dir, variable=None, lead_times=[24, 48, 72]):
    """Compute CMA for all forecast models for a specific variable at multiple lead times."""
    output_dir = os.path.join(input_dir, 'cma_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # If variable not specified, try to auto-detect from available files
    if variable is None:
        obs_files = glob.glob(os.path.join(input_dir, 'era5_obs_*.zarr'))
        if not obs_files:
            raise ValueError("No ERA5 observation files found. Please specify variable explicitly.")
        
        # Use the first observation file to determine variable
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
        #print(f"COMPUTING CMA FOR LEAD TIME: {lead_time_hours} HOURS")
        #print(f"{'='*80}\n")
        
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
                #print(f"\nProcessing forecast: {pattern}")
                forecast = open_fct(fct_path)
                
                # Store forecast initialization times for climatology processing
                if forecast_init_times is None:
                    forecast_init_times = forecast.time.values
                
                cma_vals = cma_per_lat(forecast, obs, variable, lead_time_hours)
                model_name = get_model_name_from_path(pattern)
                
                # Save results
                cma_file = os.path.join(output_dir, f'cma_{model_name}_{variable}_lead{lead_time_hours}h.txt')
                
                np.savetxt(cma_file, cma_vals)
                
                print(f"Saved CMA results to: {cma_file}")
            else:
                print(f"Warning: Forecast file not found: {fct_path}")

        # Process climatology for this lead time
        clim_path = os.path.join(input_dir, f'clim_{variable}_{lead_time_hours}h_2020.zarr')
        if os.path.exists(clim_path):
            #print(f"\nProcessing climatology: {clim_path} at {lead_time_hours}h lead time")
            climatology = open_fct(clim_path)
            
            if forecast_init_times is None:
                raise ValueError("No forecast files were processed successfully. Cannot determine forecast times for climatology.")
            
            cma_vals = clim_cma(climatology, obs, forecast_init_times, variable, lead_time_hours)
            
            # Save climatology results
            cma_file = os.path.join(output_dir, f'cma_climatology_{variable}_lead{lead_time_hours}h.txt')
            
            np.savetxt(cma_file, cma_vals)
            
            print(f"Saved climatology CMA results to: {cma_file}")
        else:
            print(f"Warning: Climatology file not found: {clim_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute CMA and/or CID (cindx) for weather forecasts at multiple lead times"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./fct_data/",
        help="Input directory containing the forecast and observation data",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="2m_temperature",
        help="Variable to process (or auto-detect from era5_obs_*.zarr if omitted).",
    )
    parser.add_argument(
        "--lead_times",
        type=int,
        nargs="+",
        default=[24, 72],
        help="Lead times in hours",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cma", "cindx", "both"],
        default="cma",
        help="Which metric(s) to compute (default: cma only).",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.rstrip("/")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if args.metric in ("cma", "both"):
        compute_cma(input_dir, args.variable, args.lead_times)
    if args.metric in ("cindx", "both"):
        compute_cindx(input_dir, args.variable, tuple(args.lead_times))
    return 0

if __name__ == "__main__":
    sys.exit(main())
