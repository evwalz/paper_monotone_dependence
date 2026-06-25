# Example figures

Standalone illustrations of monotone dependence (no external datasets).

## Dependencies

From the repo root (or this folder):

```bash
pip install numpy matplotlib seaborn pandas
pip install "git+https://github.com/evwalz/acor-python.git"
```

## Figure 1 — AGC / AKC vs discretization level k

Synthetic features at **r ∈ {0.9, 0.6, 0.3}**; outcome **Y** discretized into **2^k** equal-probability bins (**k = 1, …, 20**). Curves use **`acor(..., method="agc")`** and **`method="akc"`**; dotted lines are Spearman and Kendall references under bivariate normality.

```bash
cd example
python plot_agc_akc_discretization.py
```

Writes **`agc_akc_discretization_k.pdf`** only (scores stay in memory).

With **`n = 2^20`**, the script performs 120 **`acor`** calls and may take a few minutes.

## Figure A.1 — Triangle

```bash
cd example
python visualization.py
```

Writes **`triangle_visualization.pdf`**. Subpanels are labeled **a)**, **b)**, **c)**.

Requires: `numpy`, `matplotlib`, `seaborn` only.
