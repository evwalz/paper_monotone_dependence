#!/bin/bash

python3.8 download_data.py --lead_times 24 --variable total_precipitation_24hr

python3.8 compute_cma.py --lead_times 24 --variable total_precipitation_24hr
python3.8 compute_rmse.py --lead_times 24 --variable total_precipitation_24hr
python3.8 compute_acc.py --lead_times 24 --variable total_precipitation_24hr
python3.8 compute_seeps.py --lead_times 24 --variable total_precipitation_24hr

python3.8 plot_precipitation.py --input_dir ./fct_data/ --output_dir ./plots/ --skill_scores