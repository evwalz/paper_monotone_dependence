# LLM case study

Evaluation of large language model calibration using rank-based metrics. This folder sits next to the external [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration) repository: you use their pipeline for scores and calibration JSONs, then run the scripts here for bootstrap bundles, **CMA** / **CID** inference (Python **acor**), PDF tables, and a scatter summary.

**Stack:** everything here is **Python**. The only statistical library you must install yourself for CMA/CID is the **`acor`** package ([acor-python](https://github.com/evwalz/acor-python)). **R is not used.**

## Python dependencies (this folder)

From `case_study_LLM/`:

```bash
pip install -r requirements.txt
```

That pins **acor** (from [acor-python](https://github.com/evwalz/acor-python) on GitHub until a stable PyPI install works for you) plus **numpy**, **pandas**, **tqdm**, **matplotlib**, **seaborn**, and **scipy** used by the table and plot scripts. **Rank-Calibration** is still required separately for calibration JSONs and `from metrics import calibration` — install its `requirements.txt` in the same environment, then **export** **`RANK_CALIBRATION_PATH`** (see below) before `run_*.sh`.

`compute_table.py` and (via a shared helper) `compute_calibration_bootstrap.py` use `acor.acor_test` with the same defaults as in the acor paper package (variance `"delta"`, two-sided tests where applicable, etc.).

## Setup: Python + Rank-Calibration

1. Clone [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration) and create its environment (see their README):

   ```bash
   git clone https://github.com/shuoli90/Rank-Calibration.git
   cd Rank-Calibration
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. In that environment (or another env with the same dependencies), also **`pip install`** **`acor`** as above.

3. Point your environment at the clone: **`export RANK_CALIBRATION_PATH=/path/to/Rank-Calibration`** (the repository root). Then run **`run_table.sh`** / **`run_calibration_bootstrap.sh`**. Python adds that path to `sys.path` so `from metrics import calibration` works for **ERCE** / RCE-style scores (Python code inside Rank-Calibration, not R).

## Workflow

| Step | Command | Output |
|------|---------|--------|
| 1 | `run_calibration_bootstrap.sh` → `compute_calibration_bootstrap.py` | `*_calibration_bootstrap.json` in **`case_study_LLM/outputs/`** (20 seeds: `*_erce`, `*_cma`, `*_cindx` per uncertainty indicator; ERCE from Rank-Calibration; CMA/CID from **acor**). |
| 2 | `python plot_calibration_bootstrap.py` | **`figures/calibration_bootstrap_scatter.pdf`** — three panels: RCE (ERCE) vs CMA, RCE vs CID, CID vs CMA. |
| 3 | `run_table.sh` → `compute_table.py` | `*_cma_stat_test.json`, `*_cma_pairwise_test.json`, `*_cid_stat_test.json`, `*_cid_pairwise_test.json` in **`outputs/`** (acor `acor_test` only). |
| 4 | `python create_table.py` | **`figures/table_cma.pdf`**, **`figures/table_cid.pdf`** (with pairwise markers; reads JSON from **`outputs/`** by default). |

## Example (from this directory)

With Rank-Calibration data in place (from the same environment where you `pip install -r` this folder and Rank-Calibration):

```bash
export RANK_CALIBRATION_PATH=/path/to/Rank-Calibration   # your clone, once per shell

./run_calibration_bootstrap.sh
python plot_calibration_bootstrap.py

./run_table.sh
python create_table.py
```

- **`outputs/`** — generated JSON only (large; **gitignored** in this repo).  
- **`figures/`** — final **`table_cma.pdf`**, **`table_cid.pdf`**, and **`calibration_bootstrap_scatter.pdf`** (intended to be **committed**). Override with `--input_dir` / `--output_dir` / `--output_plot` if you split files elsewhere.

## Acknowledgments

Original calibration framework and data: [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration). Monotone dependence (CMA/CID) via [acor-python](https://github.com/evwalz/acor-python).

## Citation (Rank-Calibration)

```
Huang, X., Li, S., Yu, M., Sesia, M., Hassani, H., Lee, I., Bastani, O., 
& Dobriban, E. (2024). Uncertainty in Language Models: Assessment through 
Rank-Calibration. In Proceedings of the 2024 Conference on Empirical Methods 
in Natural Language Processing (pp. 284-312).
```
