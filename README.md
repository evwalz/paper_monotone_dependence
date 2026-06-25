# Assessing Monotone Dependence: Area Under the Curve Meets Rank Correlation

**Preprint:** [arXiv:2510.17994](https://arxiv.org/abs/2510.17994)


## Overview

This repository provides code for all empirical results in the paper. Each case study is self-contained with its own README, data sources, and dependencies.

## Prerequisites (all case studies using CMA / CID / AGC / AKC)

Install the Python [**acor**](https://github.com/evwalz/acor-python) package **once** in your environment (not listed in every `requirements.txt`):

```bash
pip install "git+https://github.com/evwalz/acor-python.git"
```

**External data** (not in this repo):

- **[WeatherBench 2](case_study_wb2/)** — zarr forecasts/obs under `case_study_wb2/fct_data/` via `download_data.py` (see WB2 README).
- **[LLM / Rank-Calibration](case_study_LLM/)** — clone [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration), then before `run_*.sh`:
  ```bash
  export RANK_CALIBRATION_PATH=/path/to/Rank-Calibration
  ```

## Repository Structure

```
paper_monotone_dependence/
├── case_study_wb2/       # WeatherBench 2 forecasting model evaluation
├── case_study_LLM/       # LLM evaluation (JSON in outputs/; paper PDFs in plots/)
├── simulation/           # Statistical hypothesis testing simulations
├── example/              # Figure 1 (discretization) + Figure A.1 (triangle)
├── README.md             # This file
├── CITATION.cff          # Machine-readable citation (arXiv:2510.17994)
└── LICENSE               # MIT License
```

## Case Studies

### [WeatherBench 2](case_study_wb2/)
Evaluation of WeatherBench 2 forecasting models (GraphCast, IFS HRES, and related baselines in the scripts) using RMSE, ACC, SEEPS, CMA, CID, and full-grid CMA/CID statistical tests.

**Data and original code**: [WeatherBench 2](https://github.com/google-research/weatherbench2)

### [LLM Evaluation](case_study_LLM/)
LLM calibration with RCE (ERCE), CMA, and CID on Rank-Calibration data; bootstrap scatter and pairwise inference tables.

**Data and original code**: [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration)

### [Simulation Studies](simulation/)
Histogram of p-value null distributions. 

### [Example](example/)
Synthetic illustrations from the paper (Figures 1 and A.1).

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

| Figure | Content | Location |
|--------|---------|----------|
| **1** | AGC/AKC vs discretization *k* | [example/](example/) |
| **A.1** | Triangle visualization | [example/](example/) |
| **2** | AGC p-value histograms (continuous DGP) | [simulation/agc/](simulation/agc/) |
| **3** | AGC p-value histograms (discrete DGP) | [simulation/agc/](simulation/agc/) |
| **4** | AKC + Zou p-value histograms | [simulation/akc/](simulation/akc/) |
| **5** | LLM calibration scatter — `plots/calibration_bootstrap_scatter.pdf` | [case_study_LLM/](case_study_LLM/) |
| **6** | WeatherBench 2 precipitation — `plots/precipitation_skill_scores_lead24h.pdf` | [case_study_wb2/](case_study_wb2/) |
| **7** | WeatherBench 2 full-grid statistical tests | [case_study_wb2/statistical_test/](case_study_wb2/statistical_test/) |
| **Table 1** | LLM pairwise CMA/CID tests | [case_study_LLM/](case_study_LLM/) |


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

This work builds upon:
- [WeatherBench 2](https://github.com/google-research/weatherbench2) for weather forecasting data
- [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration) for LLM evaluation framework