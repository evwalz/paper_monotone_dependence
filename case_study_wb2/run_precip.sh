#!/bin/bash

python download_data.py --lead_times 24 --variable total_precipitation_24hr

python compute_cma.py --lead_times 24 --variable total_precipitation_24hr
python compute_rmse.py --lead_times 24 --variable total_precipitation_24hr
python compute_acc.py --lead_times 24 --variable total_precipitation_24hr
python compute_seeps.py --lead_times 24 --variable total_precipitation_24hr

python plot_precipitation.py --input_dir ./fct_data/ --output_dir ./plots/ --skill_scores