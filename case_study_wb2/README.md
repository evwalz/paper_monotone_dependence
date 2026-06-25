# WeatherBench2 case study

Precipitation metrics and plots on [WeatherBench 2](https://github.com/google-research/weatherbench2) data (**Figure 6**). Full-grid CMA/CID tests: [`statistical_test/`](statistical_test/README.md) (**Figure 7**).

## Setup

From this folder:

```bash
pip install -r requirements.txt
pip install "git+https://github.com/evwalz/acor-python.git"   # compute_cma_cid.py
```

## Run

```bash
bash run_precip.sh
# → plots/precipitation_skill_scores_lead24h.pdf

# Or plot only (after metrics exist):
python plot_precipitation.py --input_dir ./fct_data/ --output_dir ./plots/ --skill_scores --add_panel_labels
```

- **`fct_data/`** — zarr downloads and metric `.txt` files (gitignored).
- **`plots/`** — committed paper PDFs.
