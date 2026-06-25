# Full-grid CMA/CID statistical testing (WeatherBench 2; Figure 7)

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
| `outputs/` | Grids (`.txt` / `.npz`) from `full_grid_acor_test.py`; listed in **`.gitignore`** — not committed to GitHub. |

```bash
cd case_study_wb2/statistical_test

# 1. Full-grid inference (plugin variance by default; method passed explicitly to acor_test)
python full_grid_acor_test.py --method cma
python full_grid_acor_test.py --method cid

# 2. PDFs under plots/ (reads outputs/; --method must match step 1)
python visualization.py --method cma
python visualization.py --method cid
```

`full_grid_acor_test.py` calls `acor_test(..., method=..., variance=...)` with defaults **`method=cma`** and **`variance=plugin`** (override with `--variance ij`). `visualization.py` only plots saved grids; re-run step 1 if you change variance or method.

`--check_zarr` only loads data and prints shapes (no inference). Performance of `acor` (native vs pure-Python) is determined by the installed `acor` package; see the [acor-python](https://github.com/evwalz/acor-python) README.
