# config.py - Central configuration for the pipeline
import os

# Forecast configuration
FORECAST_NAMES = ['ifs_hres', 'ifs_mean']

# Directory structure
DATA_DIR = './fct_data/'
CHUNK_DIR = './chunks'
RESULTS_DIR = './results_testing'

# Processing settings
NUM_CHUNKS = 100  # Adjust based on your needs

# File paths
OBS_PATH = os.path.join(DATA_DIR, 'era5_obs_precipitation_24hr_2020.zarr')
