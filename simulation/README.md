# Simulation 

This folder assesses the **null distribution of p-values** (histograms should be flat on \([0,1]\) under the null) for AGC and AKC.

| Subfolder | Content |
|-----------|---------|
| **`agc/`** | **Meng (1992)** on **Spearman** (reference) versus **AGC** `acor_test` (`method="agc"`) p-values.|
| **`akc/`** | **AKC** `acor_test` p-values vs. three **Zou** estimators for the same concordance **difference**. |

Shared DGP and `acor_test` alternative mapping: **`calibration_dgp.py`** (at this level).

**Naming:** In **`agc/`** and **`akc/`** the grid driver is **`run_simulation.sh`**. Default simulation output is **`results/`**; histogram PDFs go in **`plots/`** (override with **`OUTPUT_DIR`** / **`PLOTS_DIR`** env vars, or **`--output_dir`** on the Python CLIs).

## Dependencies

Install **`acor`** (not on this repo’s default path):

```bash
pip install "git+https://github.com/evwalz/acor-python.git"
```

Then from `simulation/` (or the repo root):

```bash
pip install -r requirements.txt
```

## Run (from repository root)

Set **`PYTHONPATH`** to the repo root or use the provided shell scripts.

**AGC**

```bash
cd paper_monotone_dependence   # path/to/this_repo
chmod +x simulation/agc/run_simulation.sh   # once
./simulation/agc/run_simulation.sh

python -m simulation.agc.plot_p_values \
  --results_dir simulation/agc/results \
  --output_dir simulation/agc/plots
```

**AKC**

```bash
chmod +x simulation/akc/run_simulation.sh   # once
./simulation/akc/run_simulation.sh

python -m simulation.akc.plot_p_values \
  --results_dir simulation/akc/results \
  --output_dir simulation/akc/plots
```