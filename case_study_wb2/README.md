# WeatherBench2 Case Study

A Python pipeline for evaluating weather forecast models using WeatherBench2 data.

## Quick Start

### Precipitation Analysis (24h lead time)
```bash
bash run_precip.sh
```

### All Variables (72h lead times)
```bash
bash run_multi_var.sh
```

## What It Does

1. **Downloads** forecast data from WeatherBench2 (GraphCast, IFS HRES, Persistence, Climatology)
2. **Computes** evaluation metrics (RMSE, ACC, CMA, SEEPS)
3. **Generates** plots

## Requirements

```bash
pip install xarray zarr numpy pandas scipy matplotlib seaborn
```

Python 3.8+ required.

## Supported Variables

- `2m_temperature` - 2-meter temperature
- `10m_wind_speed` - 10-meter wind speed
- `mean_sea_level_pressure` - Mean sea level pressure
- `total_precipitation_24hr` - 24-hour precipitation

## Evaluation Metrics

- **RMSE** - Root Mean Square Error
- **ACC** - Anomaly Correlation Coefficient 
- **CMA** - Coefficient of monotone association
- **SEEPS** - Stable equitable error in probability space

## Customization

Edit the shell scripts to change variables, lead times, or resolutions:

```bash
# Example: Change to 48h lead time
python download_data.py --lead_times 48 --variable 2m_temperature
python compute_rmse.py --lead_times 48 --variable 2m_temperature
```

Available options:
- `--variable`: Variable name (see list above)
- `--lead_times`: Lead times in hours (e.g., `24 48 72`)
- `--resolution`: `240x121` (default) or `64x32`
- `--input_dir`: Data directory (default: `./fct_data/`)
- `--output_dir`: Output directory (default: `./fct_data/`)

## Directory Structure

```
.
├── download_data.py          # Download data
├── compute_*.py              # Compute metrics
├── plot_*.py                 # Generate plots
├── run_*.sh                  # Quick-start scripts
└── fct_data/                 # Data and results
    ├── *.zarr                # Downloaded data
    ├── rmse_results/         # RMSE results
    ├── acc_results/          # ACC results
    ├── cma_results/          # CMA results
    └── seeps_results/        # SEEPS results
```

## Statistical Testing

The statistical testing pipeline is located in the `statistical_test/` directory. See [statistical_test/README.md](statistical_test/README.md) for detailed information about this component.

## Acknowledgments

For original code and data, see [WeatherBench 2](https://github.com/google-research/weatherbench2)

## Citation

```
Rasp, S., et al. (2023). WeatherBench 2: A benchmark for the next generation 
of data-driven global weather models. arXiv preprint arXiv:2308.15560.
```
