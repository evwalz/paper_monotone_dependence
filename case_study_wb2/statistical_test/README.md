# Statistical Testing Pipeline

This module contains scripts for performing statistical tests on weather forecast data. The pipeline is designed to handle large datasets by splitting them into chunks for parallel processing.

## Overview

The pipeline is divided into three main steps:

1. **Split Data**: Split the grid into chunks for parallel processing
2. **Process Chunks**: Process each chunk independently  
3. **Reassemble Results**: Combine results from all chunks

## Configuration

The pipeline uses a central configuration file (`config.py`) to ensure consistency across all scripts. This includes:
- Forecast names
- Directory paths
- Number of chunks

## Scripts

### 1. split_data_stat_test.py

Splits the grid data into chunks for parallel processing:
- Reads forecast and observation data from zarr files
- Standardizes dimension names
- Aligns data in time and space
- Divides the grid into chunks
- Saves chunk information and forecast names in grid_info.pkl

### 2. process_chunks.py

Processes a single chunk of the grid:
- Takes a chunk index as command-line argument
- Loads the corresponding data
- Computes statistical metrics for each grid point in the chunk
- Saves results for this chunk

### 3. chunks_reassemble.py

Combines results from all chunks:
- Loads chunk information, including forecast names from grid_info.pkl
- Reads all processed chunk files
- Reassembles the results into complete grids
- Saves the final results with the correct forecast names

## Usage

1. Edit `config.py` to set your forecast names and other parameters
2. Run the pipeline scripts in sequence:

```bash
# Step 1: Split data into chunks
python split_data_stat_test.py

# Step 2: Process chunks (can be run in parallel)
# Example: Process chunk 0
python process_chunks.py 0

# For parallel processing on multiple cores/machines
for i in {0..99}; do
    python process_chunks.py $i &
done

# Step 3: Reassemble results
python chunks_reassemble.py
```

## Dependencies

- Python 3.x
- Required Python packages:
  - numpy>=1.21.0
  - scipy>=1.7.0
  - numba>=0.54.0
  - xarray>=0.20.0
  - zarr>=2.10.0

## Notes

- The pipeline supports parallel processing for improved performance
- Results are saved in text format and can be used for further analysis
- The scripts automatically create output directories if they don't exist
- Make sure to have sufficient disk space for the chunked data 