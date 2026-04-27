# Assessing Monotone Dependence: Area Under the Curve Meets Rank Correlation

Eva-Maria Walz, Andreas Eberl, Tilmann Gneiting

**Preprint:** [arXiv:2510.17994](https://arxiv.org/abs/2510.17994)

If you use this code in your research, please cite:

```bibtex
@article{walz2025monotone,
  title={Assessing Monotone Dependence: Area Under the Curve Meets Rank Correlation},
  author={Walz, Eva-Maria and Eberl, Andreas and Gneiting, Tilmann},
  year={2025},
  eprint={2510.17994},
  archivePrefix={arXiv},
  primaryClass={stat.ME},
  url={https://arxiv.org/abs/2510.17994}
}
```

## Overview

This repository provides complete reproducibility code for all empirical results in the paper. Each case study is self-contained with its own README, data sources, and dependencies.

## Repository Structure

```
paper_monotone_dependence/
├── case_study_wb2/       # WeatherBench 2 forecasting model evaluation
├── case_study_LLM/       # LLM evaluation (JSON in outputs/; paper PDFs in figures/)
├── simulation/           # Statistical hypothesis testing simulations
├── example/              # Triangle figure for data example
├── README.md             # This file
├── CITATION.cff          # Machine-readable citation (arXiv:2510.17994)
└── LICENSE               # MIT License
```

## Case Studies

### [WeatherBench 2](case_study_wb2/)
A comprehensive evaluation of WeatherBench 2 forecasting models (GraphCast, IFS HRES, and related baselines in the scripts) using shell scripts, Python, and zarr/WeatherBench2 data: RMSE, ACC, SEEPS, CMA, and CID, plus metric plots. The core pipeline uses [`case_study_wb2/requirements.txt`](case_study_wb2/requirements.txt). **CMA/CID** (`compute_cma_cid.py`) and the optional full-grid scripts in [`case_study_wb2/statistical_test/`](case_study_wb2/statistical_test/README.md) need the **acor** package installed in addition—`pip` lines are in the case study README.

**Data and original code**: [WeatherBench 2](https://github.com/google-research/weatherbench2)

### [LLM Evaluation](case_study_LLM/)
Pure **Python** (install [**acor**](https://github.com/evwalz/acor-python) for CMA/CID; see [case_study_LLM/README.md](case_study_LLM/README.md)). Table 1: `run_table.sh` → `create_table.py` → PDFs in **`case_study_LLM/figures/`**. Bootstrap: `run_calibration_bootstrap.sh` → `plot_calibration_bootstrap.py` (scatter PDF in **`figures/`**). JSON stays in **`outputs/`** (ignored by git). Rank-Calibration supplies data and `metrics.calibration` for ERCE.

**Data and original code**: [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration)

### [Simulation Studies](simulation/)
Monte Carlo evaluation of p-value null distributions (figures 1–3 in the paper): see [`simulation/README.md`](simulation/README.md). The folder is split into [`simulation/agc/`](simulation/agc/) (**Meng** + **global AGC**) and [`simulation/akc/`](simulation/akc/) (**pairwise AKC** + **Zou**), each with its own `run_*.sh` and default `results/` directory. See the simulation README for `python3 -m simulation.agc.plot_p_values` and `python3 -m simulation.akc.plot_p_values`.

### [Example](example/)
Triangle figure illustrating monotone dependence on example data.

---

## Requirements

### General Dependencies
- Python 3.8+ (for LLM and WeatherBench 2 studies)
- Additional case-study-specific requirements listed in subdirectory READMEs

### Installation
Each case study has its own dependency file: `requirements.txt`

Navigate to the specific directory and follow the setup instructions in the local README.

## Usage

1. **Clone this repository**:
   ```bash
   git clone https://github.com/evwalz/paper_monotone_dependence.git
   cd paper_monotone_dependence
   ```

2. **Choose a case study** and navigate to its directory:
   ```bash
   cd case_study_wb2/
   ```

3. **Follow the instructions** in the case study's README to:
   - Install dependencies
   - Download/prepare data
   - Run analysis scripts
   - Generate figures/tables

## Reproducing Paper Results

Each subdirectory README contains detailed instructions for reproducing specific figures and tables from the paper:

- **Figure A1**: See [example/](example/)
- **Figures 1-3**: See [simulation/](simulation/)
- **Figure 4 & Table 1**: See [case_study_LLM/](case_study_LLM/)
- **Figure 5-6**: See [case_study_wb2/](case_study_wb2/)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

Use the BibTeX block at the top of this README, or the same reference via [arXiv:2510.17994](https://arxiv.org/abs/2510.17994).

## Acknowledgments

This work builds upon:
- [WeatherBench 2](https://github.com/google-research/weatherbench2) for weather forecasting data
- [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration) for LLM evaluation framework