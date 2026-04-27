#!/usr/bin/env bash
set -euo pipefail

python download_data.py --lead_times 72 --variable mean_sea_level_pressure
python download_data.py --lead_times 72 --variable 10m_wind_speed
python download_data.py --lead_times 72 --variable total_precipitation_24hr
python download_data.py --lead_times 72 --variable 2m_temperature

python compute_cma_cid.py --lead_times 72 --variable mean_sea_level_pressure --metric both
python compute_cma_cid.py --lead_times 72 --variable 10m_wind_speed --metric both
python compute_cma_cid.py --lead_times 72 --variable total_precipitation_24hr --metric both
python compute_cma_cid.py --lead_times 72 --variable 2m_temperature --metric both

python plot_cma_variables.py --input_dir ./fct_data/cma_results/ --output_dir ./plots/