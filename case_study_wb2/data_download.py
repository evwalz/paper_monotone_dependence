import xarray as xr
import sys
import argparse
import numpy as np

# 1. Graphcast
# 2. HRES
# 3. Ens mean
# 4. Climatology
# 5. Persistence

def compute_persistence(obs, fct_time_values):
    obs_times = obs.time.values
    #fct_time_values = forecast.time.values
    valid_times = []
    valid_target_times = []
    for t in fct_time_values:
        # For each forecast time, we need both t and t+24h to be in our obs dataset
        if t in obs_times and t + np.timedelta64(24, 'h') in obs_times:
            valid_times.append(t)
            valid_target_times.append(t + np.timedelta64(24, 'h'))

    persistence_forecast = obs.sel(time=valid_times)
    return persistence_forecast


def compute_clim(climatology, obs, time_fct):
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

    clim_forecast = clim_forecast.assign_coords(time=valid_forecast_times)
    
    return clim_forecast


def download_fct(output_dir):
    forecast_path ='gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr'
    ds = xr.open_zarr(forecast_path, decode_timedelta=True)
    ds2 = ds.sel(prediction_timedelta = ds.prediction_timedelta.values[3]).drop_vars('prediction_timedelta')
    precipitation_2020 = ds2['total_precipitation_24hr']
    

    # Create a new dataset with just this variable
    precipitation_ds_2020 = precipitation_2020.to_dataset()

    # Save locally as zarr
    local_path = output_dir + '/graphcast_ifs_precipitation_24hr_2020.zarr'
    precipitation_ds_2020.to_zarr(local_path)
    original_array = precipitation_ds_2020.time.values
 
    print(f"Saved 2020 Graphcast total_precipitation_24hr fct to {local_path}")


    ifs_data = xr.open_zarr('gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr', decode_timedelta=True)
    ifs_data2 = ifs_data.sel(prediction_timedelta = ds.prediction_timedelta.values[3]).drop_vars('prediction_timedelta')
    ifs_data3 = ifs_data2.sel(time = ds.time.values)
    # Select only the total_precipitation_24hr variable for 2020
    precipitation_2020 = ifs_data3['total_precipitation_24hr']#.sel(time=slice('2020-01-01', '2020-12-31'))

    # Create a new dataset with just this variable
    precipitation_ds_2020 = precipitation_2020.to_dataset()

    # Save locally as zarr
    local_path = output_dir + '/ifs_hres_precipitation_24hr_2020.zarr'
    precipitation_ds_2020.to_zarr(local_path)

    print(f"Saved 2020 HRES total_precipitation_24hr fct to {local_path}")


    ifs_mean = xr.open_zarr('gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr', decode_timedelta=True)
    ifs_mean2 = ifs_mean.sel(prediction_timedelta = ds.prediction_timedelta.values[3]).drop_vars('prediction_timedelta')
    ifs_mean3 = ifs_mean2.sel(time = ds.time.values)
    precipitation_2020 = ifs_mean3['total_precipitation_24hr']#.sel(time=slice('2020-01-01', '2020-12-31'))

    precipitation_ds_2020 = precipitation_2020.to_dataset()

    local_path = output_dir + '/ifs_mean_precipitation_24hr_2020.zarr'
    precipitation_ds_2020.to_zarr(local_path)

    print(f"Saved 2020 ENS mean total_precipitation_24hr fct to {local_path}")

    clim_era5 = xr.open_zarr('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr', decode_timedelta=True)
    clim_era5_2 = clim_era5.sel(hour = clim_era5.hour[[0, 2]])
    precipitation_2020 = clim_era5_2['total_precipitation_24hr']#.sel(time=slice('2020-01-01', '2020-12-31'))

    # Create a new dataset with just this variable
    precipitation_ds_2020 = precipitation_2020.to_dataset()

    # Save locally as zarr
    local_path = output_dir + '/clim_precipitation_24hr_2020.zarr'
    precipitation_ds_2020.to_zarr(local_path)


    print(f"Saved 2020 climatology total_precipitation_24hr fct to {local_path}")

    # a view more days to compute persistence forecast
    
    earliest_time = original_array[0]
    earlier_times = np.array([earliest_time - np.timedelta64(12 * (i + 1), 'h') for i in range(10)])

    # Reverse the earlier_times array so it's in chronological order
    earlier_times = earlier_times[::-1]

    # Concatenate the arrays
    combined_array = np.concatenate([earlier_times, original_array])
    era5_data = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr', decode_timedelta=True)
    era5_data3 = era5_data.sel(time = combined_array)

    precipitation_2020 = era5_data3['total_precipitation_24hr']#.sel(time=slice('2020-01-01', '2020-12-31'))

    # Create a new dataset with just this variable
    precipitation_ds_2020 = precipitation_2020.to_dataset()

    # Save locally as zarr
    local_path = output_dir + '/era5_obs_precipitation_24hr_2020.zarr'
    precipitation_ds_2020.to_zarr(local_path)

    print(f"Saved 2020 obs total_precipitation_24hr data to {local_path}")

    # also save persistence forecast:
    persistence = compute_persistence(precipitation_ds_2020, original_array)
    local_path = output_dir + '/persistence_precipitation_24hr_2020.zarr'
    persistence.to_zarr(local_path)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--output_dir', type=str, default='./fct_data/', help='Output file path')
    args = parser.parse_args()
    output_dir = args.output_dir
    download_fct(output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())