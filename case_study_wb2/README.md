# WeatherBench2 Evaluation

This repository contains scripts for evaluating forecast models from WeatherBench2 using metrics like RMSE, accuracy scores, SEEPS and CMA.

## Model Evaluation Workflow

1. First, download the required data:
```bash
python data_download.py
```

2. Then, compute the desired scores:
   - For RMSE scores:
   ```bash
   python compute_rmse.py
   ```
   - For accuracy scores:
   ```bash
   python compute_acc.py
   ```
   - For SEEPS scores:
   ```bash
   python compute_seeps.py      # Then compute SEEPS scores
   ```
   - For CMA:
   ```bash
   python compute_cma.py
   ```

3. Generate visualizations:
```bash
python visualization.py
```

## Statistical Testing Pipeline

The statistical testing pipeline is located in the `statistical_test/` directory. See the [statistical_test/README.md](statistical_test/README.md) for detailed information about this component.

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


## Acknowledgments

For original code and data, see [WeatherBench 2](https://github.com/google-research/weatherbench2)
