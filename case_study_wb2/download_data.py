import xarray as xr
import sys
import argparse
import numpy as np
import pandas as pd

def compute_persistence(obs, fct_time_values, lead_time_hours):
    """Compute persistence forecast for a specific lead time."""
    obs_times = obs.time.values
    valid_times = []
    valid_target_times = []
    lead_time_delta = np.timedelta64(lead_time_hours, 'h')
    
    for t in fct_time_values:
        # For each forecast time, we need both t and t+lead_time to be in our obs dataset
        if t in obs_times and t + lead_time_delta in obs_times:
            valid_times.append(t)
            valid_target_times.append(t + lead_time_delta)

    persistence_forecast = obs.sel(time=valid_times)
    return persistence_forecast

def get_variable_info(var_name):
    """Get variable-specific information."""
    var_info = {
        'total_precipitation_24hr': {
            'wb2_name': 'total_precipitation_24hr',
            'era5_name': 'total_precipitation_24hr',
            'is_accumulated': True,
        },
        '2m_temperature': {
            'wb2_name': '2m_temperature',
            'era5_name': '2m_temperature', 
            'is_accumulated': False,
        },
        '10m_wind_speed': {
            'wb2_name': '10m_wind_speed',
            'era5_name': '10m_wind_speed',
            'is_accumulated': False,
        },
        'mean_sea_level_pressure': {
            'wb2_name': 'mean_sea_level_pressure',
            'era5_name': 'mean_sea_level_pressure',
            'is_accumulated': False,
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

def select_lead_time(ds, lead_time_hours):
    """Select data for a specific lead time by timedelta value, not index."""
    lead_time_delta = np.timedelta64(lead_time_hours, 'h')
    
    # Select by actual timedelta value
    ds_selected = ds.sel(prediction_timedelta=lead_time_delta).drop_vars('prediction_timedelta')
    
    print(f"  Selected {lead_time_hours}h forecast using timedelta value")
    return ds_selected

def download_fct(output_dir, variable='2m_temperature', resolution='240x121', lead_time_hours=72):
    """Download forecast data for a specific lead time."""
    
    if resolution == '240x121':
        helper_name = '_with_poles_'
    else:
        helper_name = '_'
    
    # Get variable-specific information
    variable_key, var_info = get_variable_info(variable)
    wb2_name = var_info['wb2_name']
    era5_name = var_info['era5_name']

    
    # GraphCast forecast
    forecast_path = f'gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours-{resolution}_equiangular{helper_name}conservative.zarr'
    
    try:
        #print(f"\nProcessing GraphCast...")
        ds = xr.open_zarr(forecast_path, decode_timedelta=True)
        
        # Select by timedelta value, not index
        ds2 = select_lead_time(ds, lead_time_hours)
        
        if wb2_name not in ds2:
            print(f"Warning: {wb2_name} not found in GraphCast data. Available variables: {list(ds2.data_vars)}")
        else:
            var_data_2020 = ds2[wb2_name]
            var_ds_2020 = var_data_2020.to_dataset()

            # Use consistent naming with lead time
            local_path = f'{output_dir}/graphcast_ifs_{variable_key}_{lead_time_hours}h_2020.zarr'
            var_ds_2020.to_zarr(local_path, mode='w')
            original_array = var_ds_2020.time.values
            print(f"✓ Saved GraphCast {lead_time_hours}h forecast to {local_path}")
    except Exception as e:
        print(f"✗ Error processing GraphCast data: {e}")
        return

    # HRES forecast
    try:
        #print(f"\nProcessing IFS HRES...")
        ifs_data = xr.open_zarr(f'gs://weatherbench2/datasets/hres/2016-2022-0012-{resolution}_equiangular{helper_name}conservative.zarr', decode_timedelta=True)
        
        # Select by timedelta value, not index
        ifs_data2 = select_lead_time(ifs_data, lead_time_hours)
        ifs_data3 = ifs_data2.sel(time=ds.time.values)
        
        if wb2_name not in ifs_data3:
            print(f"Warning: {wb2_name} not found in HRES data. Available variables: {list(ifs_data3.data_vars)}")
        else:
            var_data_2020 = ifs_data3[wb2_name]
            var_ds_2020 = var_data_2020.to_dataset()
            local_path = f'{output_dir}/ifs_hres_{variable_key}_{lead_time_hours}h_2020.zarr'
            var_ds_2020.to_zarr(local_path, mode='w')
            print(f"✓ Saved IFS HRES {lead_time_hours}h forecast to {local_path}")
    except Exception as e:
        print(f"✗ Error processing HRES data: {e}")

    # Climatology (same for all lead times, but save separately for consistency)
    try:
        #print(f"\nProcessing Climatology...")
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
            local_path = f'{output_dir}/clim_{variable_key}_{lead_time_hours}h_2020.zarr'
            var_ds_2020.to_zarr(local_path, mode='w')
            print(f"✓ Saved climatology to {local_path}")
    except Exception as e:
        print(f"✗ Error processing climatology data: {e}")

    # ERA5 observations (download once, can be reused for all lead times)
    obs_path = f'{output_dir}/era5_obs_{variable_key}_2020.zarr'
    try:
        # Check if observations already exist
        import os
        if os.path.exists(obs_path):
            print(f"\n✓ ERA5 observations already exist at {obs_path}")
            era5_obs = xr.open_zarr(obs_path)
        else:
            #print(f"\nProcessing ERA5 observations...")
            earliest_time = original_array[0]
            # Add extra days for persistence baseline
            earlier_times = np.array([earliest_time - np.timedelta64(12 * (i + 1), 'h') for i in range(20)])
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
            
            # CRITICAL FIX: Remove encoding and rechunk before saving
            var_clean = var_ds_2020.copy()
            
            # Remove all encoding information that could cause conflicts
            for var_name in var_clean.data_vars:
                if hasattr(var_clean[var_name], 'encoding'):
                    # Clear the encoding
                    var_clean[var_name].encoding = {}
            
            # Also clear coordinate encodings
            for coord_name in var_clean.coords:
                if hasattr(var_clean.coords[coord_name], 'encoding'):
                    var_clean.coords[coord_name].encoding = {}
            
            # Rechunk to match what we want to write
            var_clean = var_clean.chunk({'time': -1, 'latitude': 240, 'longitude': 121})
            
            # Save with mode='w' to overwrite if exists
            var_clean.to_zarr(obs_path, mode='w')

            print(f"✓ Saved ERA5 observations to {obs_path}")
            era5_obs = var_clean


        # Persistence forecast for this lead time
        #print(f"\nProcessing Persistence baseline...")
        persistence = compute_persistence(era5_obs, original_array, lead_time_hours)
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
        
        local_path = f'{output_dir}/persistence_{variable_key}_{lead_time_hours}h_2020.zarr'
        persistence_clean.to_zarr(local_path, mode='w')
        print(f"✓ Saved persistence {lead_time_hours}h forecast to {local_path}")
        
    except Exception as e:
        print(f"✗ Error processing ERA5 observations: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download weather forecast data at specific lead times')
    parser.add_argument('--output_dir', type=str, default='./fct_data/', 
                       help='Output directory path')
    parser.add_argument('--resolution', type=str, default='240x121', choices=['240x121', '64x32'], 
                       help='Resolution of the data')
    parser.add_argument('--variable', type=str, default='2m_temperature', 
                       help='Variable to download. Options: 2m_temperature, 10m_wind_speed, mean_sea_level_pressure')
    parser.add_argument('--lead_times', type=int, nargs='+', default=[24],
                       help='Lead times in hours to download (e.g., --lead_times 24 48 72)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    for lead_time in args.lead_times:
        download_fct(args.output_dir, args.variable, args.resolution, lead_time)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
