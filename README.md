# Assessing Monotone Dependence: Area Under the Curve Meets Rank Correlation

<!--
**Assessing Monotone Dependence: Area Under the Curve Meets Rank Correlation**  
Eva-Maria Walz, Andreas Eberl, Tilmann Gneiting
*Preprint coming soon*
Preprint available at: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
-->
<!--
If you use this code in your research, please cite:

```bibtex
@article{walz2025monotone,
  title={Assessing Monotone Dependence: Area Under the Curve Meets Rank Correlation},
  author={Walz, Eva-Maria and Eberl, Andreas and Gneiting, Tilmann},
  year={2025},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={stat.ME},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```
-->

## Overview

This repository provides complete reproducibility code for all empirical results in the paper. Each case study is self-contained with its own README, data sources, and dependencies.

## Repository Structure

```
paper_monotone_dependence/
├── case_study_wb2/       # WeatherBench 2 forecasting model evaluation
├── case_study_LLM/       # Large Language Model evaluation
├── simulation/           # Statistical hypothesis testing simulations
├── example/              # Triangle figure for data example
├── README.md             # This file
└── LICENSE               # MIT License
```

## Case Studies

### [WeatherBench 2](case_study_wb2/)
A comprehensive evaluation of WeatherBench 2 forecasting models (GraphCast, HRES, and Ensemble forecast) using various metrics including RMSE, accuracy scores, SEEPS, and CMA. 

**Data and original code**: [WeatherBench 2](https://github.com/google-research/weatherbench2)

### [LLM Evaluation](case_study_LLM/)
A case study focusing on the evaluation of Large Language Models using monotone dependence measures.

**Data and original code**: [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration)

### [Simulation Studies](simulation/)
Simulated data for statistical hypothesis testing demonstrating properties of monotone dependence measures.

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

Citation information will be added upon publication.

## Acknowledgments

This work builds upon:
- [WeatherBench 2](https://github.com/google-research/weatherbench2) for weather forecasting data
- [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration) for LLM evaluation framework