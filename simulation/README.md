# Simulation (p-value calibration; Figures 2–4)

This folder assesses the **null distribution of p-values** (histograms should be flat on \([0,1]\) under the null) for two study types, in separate subpackages:

| Subfolder | Content |
|-----------|---------|
| **`agc/`** | **Meng (1992)** on **Spearman** (reference) + **AGC** `acor_test` (`method="agc"`) **contrast** p-value (first `pairwise_results` row; `alternative` applies). In `acor`, **CMA** is the \([0,1]\) reparametrisation of the same AGC construction on ranks; **AKC** is a different family. |
| **`akc/`** | **AKC** `acor_test` **contrast** p-value (`pairwise_results`, `method="akc"`) in ``*_our.npy``, vs. three **Zou** z-tests for the same concordance **difference**. Zou row statistics use an **O(n log n)** Fenwick path (see `akc/zou_fenwick.py` vs. R `zou_concordance_fast.cpp`). |

Shared DGP and `acor_test` alternative mapping: **`calibration_dgp.py`** (at this level).

**Naming:** In **`agc/`** and **`akc/`** the grid driver is **`run_simulation.sh`**. Default simulation output is **`results/`**; histogram PDFs go in **`plots/`** (override with **`OUTPUT_DIR`** / **`PLOTS_DIR`** env vars, or **`--output_dir`** on the Python CLIs).

**Git:** **`agc/results/`**, **`agc/plots/`**, **`akc/results/`**, and **`akc/plots/`** are listed in **`simulation/.gitignore`** and are not committed (`.npy` files and the PDF histograms produced there). Regenerate them with the shell scripts and `plot_p_values` commands below.

## Dependencies

Install **`acor`** (not on this repo’s default path):

```bash
pip install "git+https://github.com/evwalz/acor-python.git"
```

Then from `simulation/` (or the repo root):

```bash
pip install -r requirements.txt
```

**Variance:** All `acor_test` calls in this repo pass **`variance="plugin"`** explicitly (via `run_simulation.sh`, CLI defaults, and case-study scripts). The **acor** package default is **`ij`** if you omit `variance`; we do not rely on that default here.

## Run (from repository root)

Set **`PYTHONPATH`** to the repo root (the parent of the `simulation` package), or use the provided shell scripts (they do this for you).

**AGC + Meng — full paper sweep** (default `T=100000`, **`VARIANCE=plugin`**, **`ALTERNATIVE=one.sided`**, writes under `agc/results/`; discrete + continuous both one-sided):

```bash
cd paper_monotone_dependence   # path/to/this_repo
chmod +x simulation/agc/run_simulation.sh   # once
./simulation/agc/run_simulation.sh
# Faster dry run:  T=5000 ./simulation/agc/run_simulation.sh
# Custom output:   OUTPUT_DIR=./my_results VARIANCE=ij ./simulation/agc/run_simulation.sh
```

Single `n` (CLI default **`--variance plugin`**; always pass explicitly if calling `acor_test` yourself):

```bash
export PYTHONPATH="$(pwd)"
python -m simulation.agc.simulation_p_values --n 500 --T 10000 --alternative one.sided --variance plugin
python -m simulation.agc.simulation_p_values --n 500 --T 10000 --discrete --alternative one.sided --variance plugin
```

Histograms (reads `agc/results/` by default if you point `--results_dir` there; writes PDFs to `agc/plots/` with `--output_dir`). Tries **discrete/continuous** × **two.sided/one.sided** (four combinations); skips a case if the corresponding `pvals_*.npy` files are missing. Discrete runs are AGC-only (`pvals_meng` not written). Continuous figures: row **(a)** Our, **(b)** Meng.

**Paper figures (default one-sided sweep):**

| Figure | PDF (under `plots/`) |
|--------|----------------------|
| **2** | `pvalue_histograms_continuous_one_sided.pdf` |
| **3** | `pvalue_histograms_discrete_one_sided.pdf` |

```bash
python -m simulation.agc.plot_p_values \
  --results_dir simulation/agc/results \
  --output_dir simulation/agc/plots
```

- **`--alternative`** is passed to **Meng** (Spearman) and to **`acor_test`** for the **AGC** contrast p-value (first ``pairwise_results`` row). It is **not** “Meng only” anymore.

### Matching the R reference (`agc_pvals_simulation.R`)

- **`--variance`** for `acor_test` accepts **`ij`** or **`plugin`**; this repo uses **`plugin`**. The old **`delta`** option was removed upstream in the Python **acor** package.
- **IID** Python passes **`iid=True`** to `acor_test` by default, matching R **`IID = TRUE`**. For time-series / HAC (R `IID = FALSE`), pass **`--hac`**.
- **RNG**: `set.seed(42)` in R and `np.random.seed(42)` in Python do **not** draw the same normals; compare **ECDFs / histograms of p-values** instead.

---

## AKC + Zou (R `akc_pvals_sim.R`)

- **Run one job:** `python -m simulation.akc.simulation_p_values --n 500 --T 10000 --alternative one.sided` (and `--discrete` as needed). Default **`--variance plugin`**; `iid=True`; `alternative` passed through to `acor_test`. The **`_our.npy`** stream is the **contrast** p-value from ``pairwise_results`` (same rule as AGC for two predictors); Zou rows use the same row's **difference** with Zou SEs.
- **Full grid:** `chmod +x simulation/akc/run_simulation.sh` and `./simulation/akc/run_simulation.sh` (defaults: **discrete DGP**, **`VARIANCE=plugin`**, **`ALTERNATIVE=one.sided`**, **`OUTPUT_DIR=akc/results`**; override with **`VARIANCE=ij`** / **`ALTERNATIVE=two.sided`**; continuous block is commented in the script).
- **Plots (4×K panels):** `python -m simulation.akc.plot_p_values --results_dir akc/results --output_dir akc/plots` (or `--results_dir ...` / `--output_dir ...`). Row order **(a)–(d):** Our, Zou unbiased, Zou consistent, Zou simple. Produces up to **four** PDFs when data exist: **discrete/continuous** × **two.sided/one.sided** (file prefix `akc_zou_{discrete|continuous}_{two_sided|one_sided}_n{n}`). Missing `n` columns show as empty panels.

**Paper figure (default discrete one-sided sweep):** **Figure 4** → `plots/pvalue_histograms_akc_zou_discrete_one_sided.pdf`.

**Output** (per `n` and DGP), same stem as R’s `akc_zou_{discrete|continuous}_{alt}_n{n}`:

| Suffix | Contents |
|--------|----------|
| `_our.npy` | **Contrast** p-value: first ``pairwise_results`` entry from `acor_test(..., method="akc")` |
| `_zou_simple.npy`, `_zou_unbiased.npy`, `_zou_consistent.npy` | Zou z-test p-values |

## Output file names (AGC)

| Pattern | Contents |
|--------|----------|
| `pvals_agc_{...}_{alt}_n{n}.npy` | AGC **contrast** p-values from `acor_test(..., method="agc")` (first ``pairwise_results`` row) |
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
    results/            # default run_simulation.sh output (gitignored)
    plots/              # plot_p_values --output_dir (gitignored)
  akc/
    run_simulation.sh   # same name as in agc/
    helpers.py          # AKC+Zou Monte Carlo (mirrors agc/helpers.py)
    zou_concordance.py
    zou_fenwick.py
    zou_numba.py
    simulation_p_values.py
    plot_p_values.py
    results/            # default run_simulation.sh output (gitignored)
    plots/              # plot_p_values --output_dir (gitignored)
```
