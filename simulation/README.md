# Simulation Case Study

This study runs statistical siginificance tests on simulated data

## Usage

1. Run simulations to compute p-values:
```bash
bash run_simulation.sh
```

2. Generate histogram visualizations:
```bash
bash plot_simulation.sh
```

Results are saved in the `results/` directory.

## Requirements

- Python 3.x
- numpy>=1.21.0
- scipy>=1.7.0
- matplotlib>=3.4.0
- pandas>=1.3.0
- numba>=0.54.0
- tqdm>=4.60.0