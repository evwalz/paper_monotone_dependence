import xarray as xr
import zarr
import numpy as np
import pandas as pd

from typing import Callable, Union, Optional
import time
import os

# Get precipitation data:

#gsutil -m cp -r \
#  "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr/total_precipitation_24hr" \
#  ./precip_data/
# Open the Zarr array (this doesn't load data into memory yet)

z_array = zarr.open('./precip_data/total_precipitation_24hr')

# Get the shape of the array to understand its dimensions
shape = z_array.shape
print(f"Zarr array shape: {shape}")

# Create coordinate arrays
lats = np.linspace(-90, 90, 721)
lons = np.arange(0, 360, 0.25)
datetime_index = pd.date_range('1959-01-01', periods=93544, freq='6h')

# Find the index corresponding to 1990-01-01
start_idx = np.where(datetime_index >= np.datetime64('1990-01-01'))[0][0]
print(f"Starting index for 1990-01-01: {start_idx}")

# Process and save in chunks
chunk_size = 500  # Adjust based on your memory constraints
time_dim = shape[0]  # Assuming time is the first dimension

# Create temp directory for chunk files
temp_dir = './temp_netcdf_chunks'
os.makedirs(temp_dir, exist_ok=True)

# Process in time chunks
temp_files = []
for i in range(start_idx, time_dim, chunk_size):
    end_idx = min(i + chunk_size, time_dim)
    print(f"Processing time steps {i} to {end_idx} ({datetime_index[i]} to {datetime_index[end_idx-1]})")
    
    # Extract just this chunk of data
    chunk_data = z_array[i:end_idx, :, :]
    
    # Create a dataset for just this chunk
    chunk_ds = xr.Dataset(
        data_vars={
            'total_precipitation_24hr': (
                ['time', 'latitude', 'longitude'],
                chunk_data,
                {'long_name': 'Total precipitation 24hr'}
            )
        },
        coords={
            'time': datetime_index[i:end_idx],
            'latitude': lats,
            'longitude': lons
        }
    )
    
    # Write this chunk to a temporary file
    temp_file = f"{temp_dir}/chunk_{i}_{end_idx}.nc"
    chunk_ds.to_netcdf(temp_file)
    temp_files.append(temp_file)
    
    # Clear memory
    del chunk_data
    del chunk_ds

print(f"Finished creating {len(temp_files)} chunk files")

# Now combine all chunks into a single dataset using open_mfdataset
# This uses dask for lazy loading and will combine along the time dimension
print("Combining chunks into final NetCDF file...")
combined_ds = xr.open_mfdataset(temp_files, combine='by_coords')

# Write the combined dataset to a single NetCDF file
output_file = './precip_data/tp_24hr_from_1990.nc'
combined_ds.to_netcdf(output_file)
print(f"Successfully created {output_file}")

# Clean up temporary files
print("Cleaning up temporary files...")
for temp_file in temp_files:
    os.remove(temp_file)
os.rmdir(temp_dir)
print("Done!")


# and than run script to compute seeps_clim:


# dask?
#from dask.distributed import Client
#from dask import delayed

def create_window_weights(window_size: int) -> xr.DataArray:
  """Create linearly decaying window weights."""
  assert window_size % 2 == 1, 'Window size must be odd.'
  half_window_size = window_size // 2
  window_weights = np.concatenate(
      [
          np.linspace(0, 1, half_window_size + 1),
          np.linspace(1, 0, half_window_size + 1)[1:],
      ]
  )
  window_weights = window_weights / window_weights.mean()
  window_weights = xr.DataArray(window_weights, dims=['window'])
  return window_weights

def replace_time_with_doy(ds: xr.Dataset) -> xr.Dataset:
  """Replace time coordinate with days of year."""
  return ds.assign_coords({'time': ds.time.dt.dayofyear}).rename(
      {'time': 'dayofyear'}
  )

def compute_rolling_stat(
    ds: xr.Dataset,
    window_weights: xr.DataArray,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = 'mean',
) -> xr.Dataset:
  """Compute rolling climatology."""
  window_size = len(window_weights)
  half_window_size = window_size // 2  # For padding
  # Stack years
  stacked = xr.concat(
      [
          replace_time_with_doy(ds.sel(time=str(y)))
          for y in np.unique(ds.time.dt.year)
      ],
      dim='year',
  )
  # Fill gap day (366) with values from previous day 365
  stacked = stacked.fillna(stacked.sel(dayofyear=365))
  # Pad edges for perioding window
  stacked = stacked.pad(pad_width={'dayofyear': half_window_size}, mode='wrap')
  # Weighted rolling mean
  stacked = stacked.rolling(dayofyear=window_size, center=True).construct(
      'window'
  )
  if stat_fn == 'mean':
    rolling_stat = stacked.weighted(window_weights).mean(dim=('window', 'year'))
  elif stat_fn == 'std':
    rolling_stat = stacked.weighted(window_weights).std(dim=('window', 'year'))
  else:
    rolling_stat = stat_fn(
        stacked, weights=window_weights, dim=('window', 'year')
    )
  # Remove edges
  rolling_stat = rolling_stat.isel(
      dayofyear=slice(half_window_size, -half_window_size)
  )
  return rolling_stat

class SEEPSThreshold:
  """Compute SEEPS thresholds (heav/light) and fraction of dry grid points."""

  def __init__(self, dry_threshold_mm: float, var: str):
    self.dry_threshold_m = dry_threshold_mm / 1000.0
    self.var = var

  def compute(
      self,
      ds: xr.Dataset,
      dim: tuple[str],
      weights: Optional[xr.Dataset] = None,
  ):
    """Compute SEEPS thresholds and fraction of dry grid points."""
    ds = ds[self.var]
    is_dry = ds < self.dry_threshold_m
    dry_fraction = is_dry.mean(dim=dim)
    not_dry = ds.where(~is_dry)
    heavy_threshold = not_dry
    if weights is not None:
      heavy_threshold = heavy_threshold.weighted(
          weights
      )  # pytype: disable=wrong-arg-types
    heavy_threshold = heavy_threshold.quantile(2 / 3, dim=dim)
    out = xr.Dataset(
        {
            f'{self.var}_seeps_threshold': heavy_threshold.drop_vars('quantile'),
            f'{self.var}_seeps_dry_fraction': dry_fraction,
        }
    )  # fmt: skip
    return out


def compute_daily_stat(
    obs: xr.Dataset,
    window_size: int,
    clim_years: slice,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = 'mean',
) -> xr.Dataset:
    obs_daily = obs.sel(time=clim_years).resample(time='D').mean()
    window_weights = create_window_weights(window_size)
    daily_rolling_clim = compute_rolling_stat(obs_daily, window_weights, stat_fn)
    return daily_rolling_clim

def compute_daily_clim(    
    obs: xr.Dataset,
    window_size: int = 61,
    start_year: int = 1990,
    end_year: int = 2019,
    dry_threshold_mm: float = 0.25, 
    var: str = 'total_precipitation_24hr'
) -> xr.Dataset:
    clim_years = slice(str(start_year), str(end_year)+"-12-31")
    stat_fn = SEEPSThreshold(dry_threshold_mm, var=var).compute
    seeps_clim = compute_daily_stat(obs, window_size, clim_years, stat_fn)
    return seeps_clim

# Your existing functions here (create_window_weights, compute_rolling_stat, etc.)

def process_seeps_in_chunks(
    obs: xr.Dataset,
    window_size: int = 61,
    start_year: int = 1990,
    end_year: int = 2019,
    dry_threshold_mm: float = 0.25,
    var: str = 'total_precipitation_24hr',
    chunk_size: int = 50,
    output_path: str = 'seeps_climatology.nc'
):
    """Process SEEPS climatology in spatial chunks for better performance.
    
    Args:
        obs: Dataset containing precipitation data
        window_size: Size of rolling window (must be odd)
        start_year: Start year for climatology
        end_year: End year for climatology
        dry_threshold_mm: Threshold in mm below which precipitation is considered dry
        var: Name of precipitation variable
        chunk_size: Size of spatial chunks to process at once
        output_path: Path to save the final combined result
    """
    # Get the dimensions
    nlat, nlon = len(obs.latitude), len(obs.longitude)
    
    # Calculate number of chunks
    n_lat_chunks = int(np.ceil(nlat / chunk_size))
    n_lon_chunks = int(np.ceil(nlon / chunk_size))
    total_chunks = n_lat_chunks * n_lon_chunks
    
    print(f"Processing {nlat}x{nlon} grid in {n_lat_chunks}x{n_lon_chunks} chunks ({total_chunks} total chunks)")
    
    # Create a directory for temporary files
    temp_dir = 'temp_seeps_chunks'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize progress tracking
    start_time = time.time()
    chunk_count = 0
    
    # Process each chunk
    for lat_chunk in range(n_lat_chunks):
        lat_start = lat_chunk * chunk_size
        lat_end = min(lat_start + chunk_size, nlat)
        
        for lon_chunk in range(n_lon_chunks):
            lon_start = lon_chunk * chunk_size
            lon_end = min(lon_start + chunk_size, nlon)
            
            chunk_count += 1
            chunk_filename = f"{temp_dir}/chunk_{lat_chunk}_{lon_chunk}.nc"
            
            # Skip if this chunk has already been processed
            if os.path.exists(chunk_filename):
                print(f"Chunk {chunk_count}/{total_chunks} already processed, skipping")
                continue
            
            print(f"Processing chunk {chunk_count}/{total_chunks}: Lat {lat_start}:{lat_end}, Lon {lon_start}:{lon_end}")
            
            # Select the chunk from the dataset
            chunk_obs = obs.isel(
                latitude=slice(lat_start, lat_end),
                longitude=slice(lon_start, lon_end)
            )
            
            # Process this chunk
            try:
                chunk_start = time.time()
                chunk_result = compute_daily_clim(
                    obs=chunk_obs,
                    window_size=window_size,
                    start_year=start_year,
                    end_year=end_year,
                    dry_threshold_mm=dry_threshold_mm,
                    var=var
                )
                chunk_time = time.time() - chunk_start
                
                # Save the chunk result
                chunk_result.to_netcdf(chunk_filename)
                
                # Report progress
                elapsed = time.time() - start_time
                estimated_total = elapsed / chunk_count * total_chunks
                remaining = estimated_total - elapsed
                
                print(f"Chunk completed in {chunk_time:.1f}s. Overall progress: {chunk_count}/{total_chunks} chunks")
                print(f"Elapsed: {elapsed/60:.1f}m, Est. remaining: {remaining/60:.1f}m")
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
                # Continue with next chunk instead of failing completely
    
    # Combine all chunks into the final result
    print("Combining chunks into final result...")
    chunks_to_combine = []
    
    # Load all processed chunks
    for lat_chunk in range(n_lat_chunks):
        for lon_chunk in range(n_lon_chunks):
            chunk_filename = f"{temp_dir}/chunk_{lat_chunk}_{lon_chunk}.nc"
            if os.path.exists(chunk_filename):
                chunk_data = xr.open_dataset(chunk_filename)
                chunks_to_combine.append(chunk_data)
    
    # Combine chunks
    if chunks_to_combine:
        combined_result = xr.combine_by_coords(chunks_to_combine)
        
        # Save final result
        combined_result.to_netcdf(output_path)
        print(f"SEEPS climatology saved to {output_path}")
        
        # Clean up temporary files (optional)
        # for filename in os.listdir(temp_dir):
        #     os.remove(os.path.join(temp_dir, filename))
        # os.rmdir(temp_dir)
        
        return combined_result
    else:
        print("No chunks were successfully processed")
        return None


obs = xr.open_dataset('./precip_data/tp_24hr_from_1990.nc')

full_climatology = process_seeps_in_chunks(
    obs=obs,
    window_size=61,
    start_year=1990,
    end_year=2019,
    dry_threshold_mm=0.25,
    var='total_precipitation_24hr',
    chunk_size=100,
    output_path='./precip_data/global_seeps_climatology.nc'
)