# WeatherBench2 Case Study

A Python pipeline for evaluating weather forecast models using WeatherBench2 data.

## Quick Start

Install **`acor`** before running the default shell scripts (they call `compute_cma_cid.py`); see [Acor install](#acor-install) below.

### Precipitation (24h lead): download → metrics → precip plots

[`run_precip.sh`](run_precip.sh) runs `download_data`, **`compute_cma_cid`** (`--metric both`), **RMSE, ACC, SEEPS**, then **`plot_precipitation`** (five-metric PDFs under `./plots/`).

```bash
bash run_precip.sh
```

### Four variables (72h lead): download → CMA/CID only → CMA strip plots

[`run_multi_var.sh`](run_multi_var.sh) runs **`download_data`** for four variables, **`compute_cma_cid`** (`--metric both`) for each, then **`plot_cma_variables`** (reads **`cma_results/`** only). It does **not** run RMSE/ACC/SEEPS or `plot_precipitation`; add those commands yourself if you need them for these variables.

```bash
bash run_multi_var.sh
```

**Git:** The directory **`fct_data/`** is for local downloads and computed outputs only. **`.gitignore`** excludes **`fct_data/*.zarr/`** and **`fct_data/*_results/`** (metric `.txt` trees). Nothing under `fct_data/` should be pushed to GitHub; reproduce it with the steps above. To drop large blobs that were committed earlier, rewrite history (see e.g. [`git filter-repo`](https://github.com/newren/git-filter-repo)) and force-push, or accept that old commits still contain them.

## What It Does

1. **Downloads** forecast data from WeatherBench2 (GraphCast, IFS HRES, Persistence, Climatology)
2. **Computes** evaluation metrics (RMSE, ACC, CMA, CID, SEEPS). CMA and CID in `compute_cma_cid.py` use the Python [**acor**](https://github.com/evwalz/acor-python) library (`acor(forecast, obs, method=...)`), installed separately from `requirements.txt` (see [Acor install](#acor-install) below).
3. **Generates** plots — `plot_precipitation.py` writes `precipitation_metrics_lead{N}h.pdf` and/or `precipitation_skill_scores_lead{N}h.pdf` as a **five-metric** figure (RMSE, SEEPS, ACC, CMA, **CID**) in a 2×3 layout, as long as `cindx_results/` exists (run `compute_cma_cid.py` with `--metric both` or `cindx` first). `plot_cma_variables.py` reads **`cma_results/`** only. Both plot scripts load the precomputed **`.txt`** metric files under `--input_dir`. The optional **`statistical_test/`** folder is a **separate** full-grid *hypothesis testing* path (Python `acor_test` from the installed **`acor`** package); it writes under `statistical_test/outputs/` and `statistical_test/plots/` (see [`statistical_test/README.md`](statistical_test/README.md)) and does not feed the main `plot_*.py` flow unless you copy files by hand.

## Requirements

Install from this folder (includes WeatherBench2 / zarr stack and plotting):

```bash
pip install -r requirements.txt
```

Or minimal packages for **plotting only** (if metric `.txt` files already exist):

```bash
pip install numpy matplotlib seaborn
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
- **CMA** - Coefficient of monotone association (Python `acor` package, `method="cma"`)
- **CID** - C-index (Python `acor` package, `method="cid"`; `compute_cma_cid.py --metric cindx` or `both`)
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
- `--metric` (on `compute_cma_cid.py`): `cma`, `cindx`, or `both` (default `cma`)
- `--resolution`: `240x121` (default) or `64x32`
- `--input_dir`: Data directory (default: `./fct_data/` for compute scripts; `./fct_data` for `plot_precipitation.py`)
- `--output_dir`: Metric output directory (default: `./fct_data/` for compute scripts); **`plot_*.py`** default **`./plots/`** for PDFs (override as needed)

## Acor install

**`compute_cma_cid.py`** and scripts under [`statistical_test/`](statistical_test/README.md) import the **`acor`** package; it is **not** in `pip install -r requirements.txt` above. Install it once, for example:

```bash
pip install "git+https://github.com/evwalz/acor-python.git"
```

(Or `pip install acor` from PyPI when available.) Then you can run `compute_cma_cid.py` and, if needed, the full-grid tools in [statistical_test/README.md](statistical_test/README.md) (`full_grid_acor_test.py`, `visualization.py`).

## Acknowledgments

For original code and data, see [WeatherBench 2](https://github.com/google-research/weatherbench2)

## Citation

```
Rasp, S., et al. (2023). WeatherBench 2: A benchmark for the next generation 
of data-driven global weather models. arXiv preprint arXiv:2308.15560.
```
