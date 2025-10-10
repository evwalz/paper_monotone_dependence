#!/bin/bash

python download_data.py --lead_times 72 --variable mean_sea_level_pressure
python download_data.py --lead_times 72 --variable 10m_wind_speed
python download_data.py --lead_times 72 --variable total_precipitation_24hr
python download_data.py --lead_times 72 --variable 2m_temperature

python compute_cma.py --lead_times 72 --variable mean_sea_level_pressure
python compute_cma.py --lead_times 72 --variable 10m_wind_speed
python compute_cma.py --lead_times 72 --variable total_precipitation_24hr
python compute_cma.py --lead_times 72 --variable 2m_temperature

python plot_cma_variables.py --input_dir ./fct_data/cma_results/ --output_dir ./plots/