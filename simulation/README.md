# Simulation (p-value calibration; Figures 1–3)

This folder assesses the **null distribution of p-values** (histograms should be flat on \([0,1]\) under the null) for two study types, in separate subpackages:

| Subfolder | Content |
|-----------|---------|
| **`agc/`** | **Meng (1992)** on **Spearman** correlations (reference) + **global AGC** from `acor.acor_test` with `method="agc"` (two predictors, one outcome). In `acor`, **CMA** is the \([0,1]\) reparametrisation of the same AGC construction on ranks; **AKC** is a different family. |
| **`akc/`** | **Pairwise AKC** (`acor_test`, `method="akc"`) vs. three **Zou (2000)**-style z-tests for the concordance difference, matching `akc_pvals_sim.R`. Zou row statistics use an **O(n log n)** Fenwick path (see `akc/zou_fenwick.py` vs. R `zou_concordance_fast.cpp`). |

Shared DGP and `acor_test` alternative mapping: **`calibration_dgp.py`** (at this level).

**Naming:** In both **`agc/`** and **`akc/`** the grid driver is **`run_simulation.sh`**, and the default output directory is **`results/`** inside that folder (so `agc/results/` vs `akc/results/` — no name collision).

## Dependencies

Install **`acor`** (not on this repo’s default path):

```bash
pip install "git+https://github.com/evwalz/acor-python.git"
```

Then from `simulation/` (or the repo root):

```bash
pip install -r requirements.txt
```

## Run (from repository root: `pure_python/`)

Set **`PYTHONPATH`** to the repo root (the parent of the `simulation` package), or use the provided shell scripts (they do this for you).

**AGC + Meng — full paper sweep** (default `T=100000`, writes under `agc/results/` by default):

```bash
cd pure_python   # or: path/to/this_repo
chmod +x simulation/agc/run_simulation.sh   # once
./simulation/agc/run_simulation.sh
# Faster dry run:  T=5000 ./simulation/agc/run_simulation.sh
# Custom output:   OUTPUT_DIR=./my_results ./simulation/agc/run_simulation.sh
```

Single `n` (writes `pvals_*.npy` under `agc/results/` unless `--output_dir` is set):

```bash
export PYTHONPATH="$(pwd)"
python3 -m simulation.agc.simulation_p_values --n 500 --T 10000 --alternative one.sided
python3 -m simulation.agc.simulation_p_values --n 500 --T 10000 --discrete --alternative two.sided
```

Histograms (default reads `agc/results/`). Tries **discrete/continuous** × **two.sided/one.sided** (four combinations); skips a case if the corresponding `pvals_*.npy` files are missing. Discrete runs are AGC-only (`pvals_meng` not written).

```bash
python3 -m simulation.agc.plot_p_values
```

- **`--alternative` applies to Meng (Spearman) only.** The R `agc_pvals_simulation.R` script does not pass `alternative` into `acor.test` for the AGC p-value; the Python driver always uses `acor_test(..., alternative="two.sided", ...)` for AGC, while `--alternative one.sided` still drives the Meng p-value on a one-sided null.

### Matching the R reference (`agc_pvals_simulation.R`)

- **`--variance delta`** (default) matches the **package default** in `acor::acor.test`. If you set `variance = "ij"` in R’s `run_simulation_meng_our` loop, add **`--variance ij`** in Python to mirror that.
- **IID** Python passes **`iid=True`** to `acor_test` by default, matching R **`IID = TRUE`**. For time-series / HAC (R `IID = FALSE`), pass **`--hac`**.
- **RNG**: `set.seed(42)` in R and `np.random.seed(42)` in Python do **not** draw the same normals; compare **ECDFs / histograms of p-values** instead.

---

## AKC + Zou (R `akc_pvals_sim.R`)

- **Run one job:** `python3 -m simulation.akc.simulation_p_values --n 500 --T 10000 --alternative one.sided` (and `--discrete` as needed). Defaults: `variance="ij"`, `iid=True`, `alternative` passed through to `acor_test`.
- **Full grid:** `chmod +x simulation/akc/run_simulation.sh` and `./simulation/akc/run_simulation.sh` (or set `T` / `OUTPUT_DIR`).
- **Plots (4×K panels):** `python3 -m simulation.akc.plot_p_values` (default: `akc/results/`) or `--results_dir ...`. Produces up to **four** PDFs when data exist: **discrete/continuous** × **two.sided/one.sided** (file prefix `akc_zou_{discrete|continuous}_{two_sided|one_sided}_n{n}`). Missing `n` columns show as empty panels.

**Output** (per `n` and DGP), same stem as R’s `akc_zou_{discrete|continuous}_{alt}_n{n}`:

| Suffix | Contents |
|--------|----------|
| `_our.npy` | `pairwise_results` p-value (AKC difference) |
| `_zou_simple.npy`, `_zou_unbiased.npy`, `_zou_consistent.npy` | Zou z-test p-values |

## Output file names (AGC)

| Pattern | Contents |
|--------|----------|
| `pvals_agc_{...}_{alt}_n{n}.npy` | Global p-values from `acor_test(..., method="agc")` |
| `pvals_meng_...` (continuous only) | Meng test on Spearman corrs |

## Layout

```
simulation/
  calibration_dgp.py    # shared DGP + alternative mapping
  requirements.txt
  agc/
    run_simulation.sh   # AGC full-grid sweep
    helpers.py
    simulation_p_values.py
    plot_p_values.py
    results/            # default --output_dir for AGC (created on run)
  akc/
    run_simulation.sh   # same name as in agc/
    helpers.py          # AKC+Zou Monte Carlo (mirrors agc/helpers.py)
    zou_concordance.py
    zou_fenwick.py
    zou_numba.py
    simulation_p_values.py
    plot_p_values.py
    results/            # default --output_dir (mirrors agc/results/)
```
