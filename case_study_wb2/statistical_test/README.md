# Full-grid CMA/CID tests (Figure 7)

Pairwise **CMA** / **CID** inference on a lat–lon grid. Requires [**acor**](https://github.com/evwalz/acor-python) and zarr data (paths in `config.py`).

## Run

```bash
cd case_study_wb2/statistical_test

python full_grid_acor_test.py --method cma
python full_grid_acor_test.py --method cid

python visualization.py --method cma    # → plots/p_vals_cma_*.pdf
python visualization.py --method cid    # → plots/p_vals_cid_*.pdf
```

- **`outputs/`** — gridded results from `full_grid_acor_test.py` (gitignored).
- **`plots/`** — committed paper PDFs.
