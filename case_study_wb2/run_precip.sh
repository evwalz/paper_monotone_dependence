#!/usr/bin/env bash
set -euo pipefail

python download_data.py --lead_times 24 --variable total_precipitation_24hr

python compute_cma_cid.py --lead_times 24 --variable total_precipitation_24hr --metric both
python compute_rmse.py --lead_times 24 --variable total_precipitation_24hr
python compute_acc.py --lead_times 24 --variable total_precipitation_24hr
python compute_seeps.py --lead_times 24 --variable total_precipitation_24hr

python plot_precipitation.py --input_dir ./fct_data/ --output_dir ./plots/ --skill_scores
python plot_precipitation.py --input_dir ./fct_data/ --output_dir ./plots/