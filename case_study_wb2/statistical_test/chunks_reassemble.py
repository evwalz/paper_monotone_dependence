#!/usr/bin/env python
# reassemble_results.py - Script to combine results from all chunks

import numpy as np
import os
import glob
import pickle
from pathlib import Path
from config import CHUNK_DIR, RESULTS_DIR

def reassemble_results(chunk_dir=CHUNK_DIR, result_dir=RESULTS_DIR, output_dir=RESULTS_DIR):
    """
    Reassemble results from all processed chunks.
    
    Parameters:
    -----------
    chunk_dir : str
        Directory where chunk info is stored
    result_dir : str
        Directory where chunk results are stored
    output_dir : str
        Directory to save final results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load grid information
    with open(f'{chunk_dir}/grid_info.pkl', 'rb') as f:
        grid_info = pickle.load(f)
    
    n_lat = grid_info['n_lat']
    n_lon = grid_info['n_lon']
    
    # Get forecast names from grid_info, ensuring consistency with split_data_stat_test.py
    forecast_name = grid_info.get('forecast_names', ['ifs_hres', 'ifs_mean'])  # Fallback if not found
    print(f"Using forecast names from grid_info: {forecast_name}")
    
    # Initialize result arrays
    cma_fct1 = np.zeros((n_lat, n_lon))
    cma_fct2 = np.zeros((n_lat, n_lon))
    p_fct1 = np.zeros((n_lat, n_lon))
    p_fct2 = np.zeros((n_lat, n_lon))
    ci_lower = np.zeros((n_lat, n_lon))
    ci_upper = np.zeros((n_lat, n_lon))
    global_grid = np.zeros((n_lat, n_lon))
    variance_grid = np.zeros((n_lat, n_lon))
    
    # Fill arrays with NaN to identify missing points
    cma_fct1.fill(np.nan)
    cma_fct2.fill(np.nan)
    p_fct1.fill(np.nan)
    p_fct2.fill(np.nan)
    ci_lower.fill(np.nan)
    ci_upper.fill(np.nan)
    global_grid.fill(np.nan)
    variance_grid.fill(np.nan)
    
    # Get all chunk result files
    chunk_files = glob.glob(f'{result_dir}/chunk_*.npz')
    
    if not chunk_files:
        print(f"Error: No chunk results found in {result_dir}")
        return
    
    print(f"Found {len(chunk_files)} chunk result files")
    
    # Process each chunk file
    for chunk_file in chunk_files:
        chunk_data = np.load(chunk_file)
        
        lat_indices = chunk_data['lat_indices']
        lon_indices = chunk_data['lon_indices']
        
        cma_fct1_chunk = chunk_data['cma_fct1']
        cma_fct2_chunk = chunk_data['cma_fct2']
        p_fct1_chunk = chunk_data['p_fct1']
        p_fct2_chunk = chunk_data['p_fct2']
        ci_lower_chunk = chunk_data['ci_lower']
        ci_upper_chunk = chunk_data['ci_upper']
        global_grid_chunk = chunk_data['global_grid']
        variance_grid_chunk = chunk_data['variance']
        
        # Place chunk results in the appropriate grid locations
        for i in range(len(lat_indices)):
            lat_idx = lat_indices[i]
            lon_idx = lon_indices[i]
            
            cma_fct1[lat_idx, lon_idx] = cma_fct1_chunk[i]
            cma_fct2[lat_idx, lon_idx] = cma_fct2_chunk[i]
            p_fct1[lat_idx, lon_idx] = p_fct1_chunk[i]
            p_fct2[lat_idx, lon_idx] = p_fct2_chunk[i]
            ci_lower[lat_idx, lon_idx] = ci_lower_chunk[i]
            ci_upper[lat_idx, lon_idx] = ci_upper_chunk[i]
            global_grid[lat_idx, lon_idx] = global_grid_chunk[i]
            variance_grid[lat_idx, lon_idx] = variance_grid_chunk[i]
    
    # Check for missing values
    missing_points = np.sum(np.isnan(cma_fct1))
    if missing_points > 0:
        print(f"Warning: {missing_points} grid points have missing values")
    
    # Save reassembled results
    name_fct1 = forecast_name[0]
    name_fct2 = forecast_name[1]
    np.savetxt(f'{output_dir}/cma_{name_fct1}.txt', cma_fct1)
    np.savetxt(f'{output_dir}/cma_{name_fct2}.txt', cma_fct2)
    np.savetxt(f'{output_dir}/p_{name_fct1}.txt', p_fct1)
    np.savetxt(f'{output_dir}/p_{name_fct2}.txt', p_fct2)
    np.savetxt(f'{output_dir}/ci_lower_{name_fct1}_{name_fct2}.txt', ci_lower)
    np.savetxt(f'{output_dir}/ci_upper_{name_fct1}_{name_fct2}.txt', ci_upper)
    np.savetxt(f'{output_dir}/p_global_{name_fct1}_{name_fct2}.txt', global_grid)
    np.savetxt(f'{output_dir}/variance_{name_fct1}_{name_fct2}.txt', variance_grid)
    
    print(f"Results successfully reassembled and saved to {output_dir}")

if __name__ == "__main__":
    reassemble_results()