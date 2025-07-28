# Simulation Case Study

This study runs statistical siginificance tests.

## Main Scripts

- `simulated_data.py`: Generates synthetic data and performs statistical tests:
  - Binary data simulation
  - Real-valued data simulation
  - Discretized data simulation
- `helpers.py`: Contains utility functions

## Requirements

- Python 3.x
- Required Python packages:
  - numpy>=1.21.0
  - scipy>=1.7.0
  - xarray>=0.20.0
  - matplotlib>=3.4.0
  - pandas>=1.3.0

## Usage

Run the simulation script with the following arguments:

```bash
python simulated_data.py [--output_dir OUTPUT_DIR] [--n N] [--T T] [--experiment EXPERIMENT]
```

### Arguments:
- `--output_dir`: Output directory for saving results (default: './simulated_data/')
- `--n`: Sample size per repetition (default: 100)
- `--T`: Number of repetitions (default: 100)
- `--experiment`: Type of experiment to run:
  - `binary`: Simulates binary data with normal latent variables
  - `real`: Simulates real-valued data with normal distributions
  - `discretized`: Simulates discretized data from normal distributions
  (default: 'binary')

Example usage:
```bash
# Run binary experiment with default parameters
python simulated_data.py

# Run real experiment with custom parameters
python simulated_data.py --n 200 --T 1000 --experiment real --output_dir ./results/
```