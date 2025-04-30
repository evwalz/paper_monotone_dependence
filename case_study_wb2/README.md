# Weather Prediction Model Evaluation

This repository contains scripts for evaluating weather prediction models, including GraphCast, HRES and Ensemble forecast using metrics like RMSE, accuracy scores, SEEPS and CMA and CPA.

## Overview

The repository contains two main components:

1. **Model Evaluation Scripts**: For computing various metrics
2. **Statistical Testing Pipeline**: For performing statistical tests on forecast data (located in `statistical_test/` directory)

## Model Evaluation Workflow

1. First, download the required data:
```bash
python data_download.py
```

2. Then, compute the desired scores:
   - For RMSE scores:
   ```bash
   python compute_scores.py
   ```
   - For accuracy scores:
   ```bash
   python compute_acc.py
   ```
   - For SEEPS scores:
   ```bash
   # First download precipitation data
   gsutil -m cp -r "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr/total_precipitation_24hr" ./precip_data/
   
   # Then compute SEEPS scores
   python create_seeps_clim.py  # First create climatology
   python compute_seeps.py      # Then compute SEEPS scores
   ```
   - For CMA/CPA:
   ```bash
   python compute_cma_cpa.py
   ```

3. Generate visualizations:
```bash
python visualization.py [--input_dir <input_directory>] [--metric <metric>]
```

## Statistical Testing Pipeline

The statistical testing pipeline is located in the `statistical_test/` directory. See the [statistical_test/README.md](statistical_test/README.md) for detailed information about this component.

## Script Descriptions

### Model Evaluation Scripts
- `data_download.py`: Downloads required data for model evaluation
- `compute_rmse.py`: Computes RMSE scores for different models
- `compute_acc.py`: Computes accuracy scores for different models
- `create_seeps_clim.py`: Creates climatology data required for SEEPS score computation
- `compute_seeps.py`: Computes SEEPS (Stable Equitable Error in Probability Space) scores for different models
- `visualization.py`: Generates RMSE plots and visualizations
- `helper.py`: Contains utility functions used by other scripts
- `helper_stats.py`: Contains statistical utility functions used across scripts

### Statistical Testing Scripts
Located in `statistical_test/` directory:
- `config.py`: Central configuration for the statistical testing pipeline
- `split_data_stat_test.py`: Splits grid data for parallel processing
- `process_chunks.py`: Processes individual chunks of data
- `chunks_reassemble.py`: Combines results from all chunks

## Requirements

- Python 3.x
- Required Python packages:
  - numpy>=1.21.0
  - scipy>=1.7.0
  - numba>=0.54.0
  - xarray>=0.20.0
  - zarr>=2.10.0
  - matplotlib>=3.4.0
  - seaborn>=0.11.0
  - pandas>=1.3.0
  - scikit-learn>=0.24.0

## Notes

- The scripts use Numba for performance optimization
- Results are saved in text format and can be used for further analysis
- The scripts automatically create output directories if they don't exist
- `helper.py` provides essential functions used across other scripts
- The statistical testing pipeline supports parallel processing for improved performance 