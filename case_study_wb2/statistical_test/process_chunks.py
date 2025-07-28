# process_chunk.py - Script to process a single chunk
import numpy as np
import os
import pickle
import sys
from helper_stats import cma_time_series
from config import CHUNK_DIR, RESULTS_DIR

def process_chunk(chunk_idx, chunk_dir=CHUNK_DIR, output_dir=RESULTS_DIR):
    """
    Process a single chunk of the grid.
    
    Parameters:
    -----------
    chunk_idx : int
        Index of the chunk to process
    chunk_dir : str
        Directory where chunks are stored
    output_dir : str
        Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load grid information
    with open(f'{chunk_dir}/grid_info.pkl', 'rb') as f:
        grid_info = pickle.load(f)
    
    n_lat = grid_info['n_lat']
    n_lon = grid_info['n_lon']
    chunks = grid_info['chunks']
    
    if chunk_idx >= len(chunks):
        print(f"Error: Chunk index {chunk_idx} is out of range")
        return
    
    start_idx, end_idx = chunks[chunk_idx]
    
    # Load data
    fct1 = np.load(f'{chunk_dir}/fct1.npy')
    fct2 = np.load(f'{chunk_dir}/fct2.npy')
    var_obs = np.load(f'{chunk_dir}/var_obs.npy')
    
    # Initialize arrays for results
    cma_fct1_chunk = np.zeros(end_idx - start_idx)
    cma_fct2_chunk = np.zeros(end_idx - start_idx)
    p_fct1_chunk = np.zeros(end_idx - start_idx)
    p_fct2_chunk = np.zeros(end_idx - start_idx)
    ci_lower_chunk = np.zeros(end_idx - start_idx)
    ci_upper_chunk = np.zeros(end_idx - start_idx)
    global_grid_chunk = np.zeros(end_idx - start_idx)
    variance_grid_chunk = np.zeros(end_idx - start_idx)
    
    # Also store lat/lon indices for reassembly
    lat_indices = np.zeros(end_idx - start_idx, dtype=int)
    lon_indices = np.zeros(end_idx - start_idx, dtype=int)
    
    # Process each point in the chunk
    for i in range(start_idx, end_idx):
        # Convert flat index to 2D grid indices
        lat_idx = i // n_lon
        lon_idx = i % n_lon
        
        # Store indices for reassembly
        chunk_i = i - start_idx
        lat_indices[chunk_i] = lat_idx
        lon_indices[chunk_i] = lon_idx
        
        # Extract time series for this grid point
        fct1_series = fct1[:, lat_idx, lon_idx]
        fct2_series = fct2[:, lat_idx, lon_idx]
        obs_series = var_obs[:, lat_idx, lon_idx]
        
        # Compute CMA metrics
        x0 = np.vstack((fct1_series, fct2_series))
        try:
            cmas, pvals, ci, global_p, variance = cma_time_series(obs_series, x0)
            cma_fct1_chunk[chunk_i] = cmas[0]
            cma_fct2_chunk[chunk_i] = cmas[1]
            p_fct1_chunk[chunk_i] = pvals[0]
            p_fct2_chunk[chunk_i] = pvals[1]
            ci_lower_chunk[chunk_i] = ci[0]
            ci_upper_chunk[chunk_i] = ci[1]
            global_grid_chunk[chunk_i] = global_p
            variance_grid_chunk[chunk_i] = variance
        except Exception as e:
            print(f"Error at lat {lat_idx}, lon {lon_idx}: {e}")
            # Set to NaN
            cma_fct1_chunk[chunk_i] = np.nan
            cma_fct2_chunk[chunk_i] = np.nan
            p_fct1_chunk[chunk_i] = np.nan
            p_fct2_chunk[chunk_i] = np.nan
            ci_lower_chunk[chunk_i] = np.nan
            ci_upper_chunk[chunk_i] = np.nan
            global_grid_chunk[chunk_i] = np.nan
            variance_grid_chunk[chunk_i] = np.nan
    
    # Save results for this chunk
    np.savez(
        f'{output_dir}/chunk_{chunk_idx}.npz',
        lat_indices=lat_indices,
        lon_indices=lon_indices,
        cma_fct1=cma_fct1_chunk,
        cma_fct2=cma_fct2_chunk,
        p_fct1=p_fct1_chunk,
        p_fct2=p_fct2_chunk,
        ci_lower=ci_lower_chunk,
        ci_upper=ci_upper_chunk,
        global_grid=global_grid_chunk,
        variance = variance_grid_chunk
    )
    
    #print(f"Processed chunk {chunk_idx}, points {start_idx} to {end_idx}")

if __name__ == "__main__":
    # Get chunk index from command line argument
    if len(sys.argv) > 1:
        chunk_idx = int(sys.argv[1])
        process_chunk(chunk_idx)
    else:
        print("Usage: python process_chunk.py CHUNK_INDEX")