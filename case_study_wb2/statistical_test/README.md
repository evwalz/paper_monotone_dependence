# Full-grid CMA/CID statistical testing (WeatherBench 2)

## Install `acor` (required)

Run this **before** any script in this folder. The case study’s [`../requirements.txt`](../requirements.txt) does not include it.

**From PyPI** (after a release is published):

```bash
pip install acor
```

**From GitHub** (works even when PyPI is not up to date):

```bash
pip install "git+https://github.com/evwalz/acor-python.git"
```

Source and build details: [acor-python](https://github.com/evwalz/acor-python).

The same **`acor`** install is required by [`../compute_cma_cid.py`](../compute_cma_cid.py) for CMA/CID metric outputs (`cma_results/`, `cindx_results/`).

---

The rest of the WeatherBench2 pipeline (without CMA/CID from `compute_cma_cid.py` and without this folder) only needs [`../requirements.txt`](../requirements.txt). This folder is optional: pairwise CMA or CID *inference* on a full lat–lon grid, plus `visualization.py` for PDF plots. You also need the usual case-study stack from [`../requirements.txt`](../requirements.txt) and zarr data (paths in `config.py`).

**Layout** (mirrors the rest of WB2: use **`outputs/`** for gridded / numeric products and **`plots/`** for PDFs, like `../plots/` in the main case study).

| File | Role |
|------|------|
| `config.py` | Forecasts, variable, lead time, `DATA_DIR`, `OUTPUTS_DIR`, `PLOTS_DIR`. |
| `full_grid_acor_test.py` | Zarr → per-cell `acor_test` (CMA or CID) → `outputs/*.txt` and `.npz`. |
| `visualization.py` | Reads grids from `outputs/`; writes PDFs to `plots/` (see `argparse`). |
| `outputs/` | Grids and small example `.txt` files (bulky `.npz` may be gitignored). |

```bash
# CMA (default) or CID: --method cma | --method cid
python full_grid_acor_test.py
python full_grid_acor_test.py --method cid
python visualization.py   # with the flags you need
```

`--check_zarr` only loads data and prints shapes (no inference). Performance of `acor` (native vs pure-Python) is determined by the installed `acor` package; see the [acor-python](https://github.com/evwalz/acor-python) README.
