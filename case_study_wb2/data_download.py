import xarray as xr
import sys
import argparse
import numpy as np
import os

def compute_persistence(obs, fct_time_values):
    obs_times = obs.time.values
    valid_times = []
    valid_target_times = []
    for t in fct_time_values:
        # For each forecast time, we need both t and t+24h to be in our obs dataset
        if t in obs_times and t + np.timedelta64(24, 'h') in obs_times:
            valid_times.append(t)
            valid_target_times.append(t + np.timedelta64(24, 'h'))

    persistence_forecast = obs.sel(time=valid_times)
    return persistence_forecast

def get_variable_info(var_name):
    """Get variable-specific information including prediction timedelta index and verification logic."""
    var_info = {
        'total_precipitation_24hr': {
            'timedelta_idx': 3,  # 24 hours (1 day)
            'wb2_name': 'total_precipitation_24hr',  # WeatherBench2 name
            'era5_name': 'total_precipitation_24hr',  # ERA5 name
            'is_accumulated': True,
            'verification_offset': 24  # hours
        },
        '2m_temperature': {
            'timedelta_idx': 3,  # 24 hours (1 day)
            'wb2_name': '2m_temperature',
            'era5_name': '2m_temperature', 
            'is_accumulated': False,
            'verification_offset': 24  # hours
        },
        '10m_wind_speed': {
            'timedelta_idx': 3,  # 24 hours (1 day)
            'wb2_name': '10m_wind_speed',
            'era5_name': '10m_wind_speed',
            'is_accumulated': False,
            'verification_offset': 24  # hours
        },
        'mean_sea_level_pressure': {
            'timedelta_idx': 3,  # 24 hours (1 day)
            'wb2_name': 'mean_sea_level_pressure',
            'era5_name': 'mean_sea_level_pressure',
            'is_accumulated': False,
            'verification_offset': 24  # hours
        }
    }
    
    # Map common aliases
    alias_map = {
        'T2M': '2m_temperature',
        't2m': '2m_temperature',
        'WS10': '10m_wind_speed',
        'ws10': '10m_wind_speed',
        'MSLP': 'mean_sea_level_pressure',
        'mslp': 'mean_sea_level_pressure',
        'precipitation': 'total_precipitation_24hr',
        'precip': 'total_precipitation_24hr'
    }
    
    # Check if it's an alias
    if var_name in alias_map:
        var_name = alias_map[var_name]
    
    if var_name not in var_info:
        raise ValueError(f"Variable {var_name} not supported. Available variables: {list(var_info.keys())}")
    
    return var_name, var_info[var_name]

def download_fct(output_dir, variable='total_precipitation_24hr', resolution='240x121'):
    if resolution == '240x121':
        helper_name = '_with_poles_'
    else:
        helper_name = '_'
    
    # Get variable-specific information
    variable_key, var_info = get_variable_info(variable)
    wb2_name = var_info['wb2_name']
    era5_name = var_info['era5_name']
    timedelta_idx = var_info['timedelta_idx']
    
    print(f"Processing variable: {variable} -> {variable_key}")
    print(f"WeatherBench2 name: {wb2_name}, ERA5 name: {era5_name}")
    print(f"Timedelta index: {timedelta_idx}, Is accumulated: {var_info['is_accumulated']}")
    
    # GraphCast forecast
    forecast_path = f'gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours-{resolution}_equiangular{helper_name}conservative.zarr'
    
    try:
        ds = xr.open_zarr(forecast_path, decode_timedelta=True)
        ds2 = ds.sel(prediction_timedelta=ds.prediction_timedelta.values[timedelta_idx]).drop_vars('prediction_timedelta')
        
        if wb2_name not in ds2:
            print(f"Warning: {wb2_name} not found in GraphCast data. Available variables: {list(ds2.data_vars)}")
        else:
            var_data_2020 = ds2[wb2_name]
            var_ds_2020 = var_data_2020.to_dataset()

            # Use consistent naming
            local_path = f'{output_dir}/graphcast_ifs_{variable_key}_2020.zarr'
            var_ds_2020.to_zarr(local_path)
            original_array = var_ds_2020.time.values
            print(f"Saved 2020 GraphCast {variable_key} forecast to {local_path}")
    except Exception as e:
        print(f"Error processing GraphCast data: {e}")
        return

    # HRES forecast
    try:
        ifs_data = xr.open_zarr(f'gs://weatherbench2/datasets/hres/2016-2022-0012-{resolution}_equiangular{helper_name}conservative.zarr', decode_timedelta=True)
        ifs_data2 = ifs_data.sel(prediction_timedelta=ds.prediction_timedelta.values[timedelta_idx]).drop_vars('prediction_timedelta')
        ifs_data3 = ifs_data2.sel(time=ds.time.values)
        
        if wb2_name not in ifs_data3:
            print(f"Warning: {wb2_name} not found in HRES data. Available variables: {list(ifs_data3.data_vars)}")
        else:
            var_data_2020 = ifs_data3[wb2_name]
            var_ds_2020 = var_data_2020.to_dataset()
            local_path = f'{output_dir}/ifs_hres_{variable_key}_2020.zarr'
            var_ds_2020.to_zarr(local_path)
            print(f"Saved 2020 HRES {variable_key} forecast to {local_path}")
    except Exception as e:
        print(f"Error processing HRES data: {e}")

    # Climatology
    try:
        clim_era5 = xr.open_zarr(f'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_{resolution}_equiangular{helper_name}conservative.zarr', decode_timedelta=True)
        clim_era5_2 = clim_era5.sel(hour=clim_era5.hour[[0, 2]])  # Select 00 and 12 UTC
        
        if era5_name not in clim_era5_2:
            print(f"Warning: {era5_name} not found in climatology data. Available variables: {list(clim_era5_2.data_vars)}")
        else:
            var_data_2020 = clim_era5_2[era5_name]
            var_ds_2020 = var_data_2020.to_dataset()
            # Rename to match our standard naming
            if era5_name != variable_key:
                var_ds_2020 = var_ds_2020.rename({era5_name: variable_key})
            local_path = f'{output_dir}/clim_{variable_key}_2020.zarr'
            var_ds_2020.to_zarr(local_path)
            print(f"Saved 2020 climatology {variable_key} to {local_path}")
    except Exception as e:
        print(f"Error processing climatology data: {e}")

    # ERA5 observations (including extra days for persistence)
    try:
        earliest_time = original_array[0]
        earlier_times = np.array([earliest_time - np.timedelta64(12 * (i + 1), 'h') for i in range(10)])
        earlier_times = earlier_times[::-1]  # Reverse to chronological order
        combined_array = np.concatenate([earlier_times, original_array])
        
        era5_data = xr.open_zarr(f'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-{resolution}_equiangular{helper_name}conservative.zarr', decode_timedelta=True)
        era5_data3 = era5_data.sel(time=combined_array)

        if era5_name not in era5_data3:
            print(f"Warning: {era5_name} not found in ERA5 observations. Available variables: {list(era5_data3.data_vars)}")
            return
        
        var_data_2020 = era5_data3[era5_name]
        var_ds_2020 = var_data_2020.to_dataset()
        
        # Rename to match our standard naming
        if era5_name != variable_key:
            var_ds_2020 = var_ds_2020.rename({era5_name: variable_key})
        
        local_path = f'{output_dir}/era5_obs_{variable_key}_2020.zarr'
        
        if resolution == '64x32':
            var_clean = var_ds_2020.copy()
            for var_name in var_clean.data_vars:
                if hasattr(var_clean[var_name], 'encoding') and 'chunks' in var_clean[var_name].encoding:
                    del var_clean[var_name].encoding['chunks']
            var_clean.to_zarr(local_path)
        else:
            var_ds_2020.to_zarr(local_path)

        print(f"Saved 2020 ERA5 observations {variable_key} to {local_path}")

        # Persistence forecast
        persistence = compute_persistence(var_ds_2020, original_array)
        persistence_rechunked = persistence.chunk({'time': -1})
        
        new_data = {}
        new_coords = {}
        
        # Copy data variables without encoding
        for var_name in persistence_rechunked.data_vars:
            data_array = persistence_rechunked[var_name]
            new_data[var_name] = (data_array.dims, data_array.data, data_array.attrs)
        
        # Copy coordinates
        for coord_name in persistence_rechunked.coords:
            coord_array = persistence_rechunked.coords[coord_name]
            new_coords[coord_name] = (coord_array.dims, coord_array.data, coord_array.attrs)
        
        # Create fresh dataset with no encoding
        persistence_clean = xr.Dataset(new_data, coords=new_coords, attrs=persistence_rechunked.attrs)
        
        local_path = f'{output_dir}/persistence_{variable_key}_2020.zarr'
        persistence_clean.to_zarr(local_path)
        print(f"Saved 2020 persistence {variable_key} forecast to {local_path}")
        
    except Exception as e:
        print(f"Error processing ERA5 observations: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download weather forecast data for various variables')
    parser.add_argument('--output_dir', type=str, default='./fct_data/', 
                       help='Output directory path')
    parser.add_argument('--resolution', type=str, default='240x121', choices=['240x121', '64x32'], help='Resolution of the data')
    parser.add_argument('--variable', type=str, default='total_precipitation_24hr', 
                       help='Variable to download. Options: total_precipitation_24hr (TP), 2m_temperature (T2M), 10m_wind_speed (WS10), mean_sea_level_pressure (MSLP)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Data will be saved to: {args.output_dir}")
    download_fct(args.output_dir, args.variable, args.resolution)
    return 0

if __name__ == "__main__":
    sys.exit(main())