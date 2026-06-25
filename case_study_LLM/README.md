# LLM case study

Bootstrap calibration and **CMA** / **CID** inference on [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration) data (**Figure 5**, **Table 1**).

## Setup

From this folder:

```bash
pip install -r requirements.txt
export RANK_CALIBRATION_PATH=/path/to/Rank-Calibration   # clone; see repo root README
```

Requires [**acor**](https://github.com/evwalz/acor-python) (in `requirements.txt`) and Rank-Calibration’s own dependencies in the same environment.

## Run

```bash
./run_calibration_bootstrap.sh
python plot_calibration_bootstrap.py    # → plots/calibration_bootstrap_scatter.pdf

./run_table.sh
python create_table.py                  # → plots/table_cma.pdf, plots/table_cid.pdf
```

- **`outputs/`** — JSON from the shell scripts (gitignored).
- **`plots/`** — committed paper PDFs.
