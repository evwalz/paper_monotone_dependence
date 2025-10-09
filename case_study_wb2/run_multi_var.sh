#!/bin/bash

python3.8 download_data.py --lead_times 72 --variable mean_sea_level_pressure
python3.8 download_data.py --lead_times 72 --variable 10m_wind_speed
python3.8 download_data.py --lead_times 72 --variable total_precipitation_24hr
python3.8 download_data.py --lead_times 72 --variable 2m_temperature

python3.8 compute_cma.py --lead_times 72 --variable mean_sea_level_pressure
python3.8 compute_cma.py --lead_times 72 --variable 10m_wind_speed
python3.8 compute_cma.py --lead_times 72 --variable total_precipitation_24hr
python3.8 compute_cma.py --lead_times 72 --variable 2m_temperature

python3.8 plot_cma_variables.py --input_dir ./fct_data/cma_results/ --output_dir ./plots/