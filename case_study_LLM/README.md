# LLM Case Study

Evaluation of Large Language Models.

## Dependencies

This case study requires the [Rank-Calibration](https://github.com/shuoli90/Rank-Calibration) repository as a dependency. Please follow these steps to set up the environment:

1. Clone the Rank-Calibration repository:
```bash
git clone https://github.com/shuoli90/Rank-Calibration.git
cd Rank-Calibration
```

2. Set up the environment as described in their README:
```bash
python -m venv rce
pip install -r requirements.txt
```

## Running the Script

To run the analysis, use the provided `run_compute_cma.sh` script which will execute `compute_cma.py` with different configurations:

1. First, make sure you're in the Rank-Calibration directory:
```bash
cd /path/to/Rank-Calibration
```

2. Activate the Rank-Calibration environment:
```bash
# For Linux/Mac
source rce/bin/activate

# For Windows (Command Prompt)
rce\Scripts\activate

# For Windows (PowerShell)
.\rce\Scripts\Activate.ps1
```

3. Navigate to the directory containing your script:
```bash
cd /path/to/case_study_LLM
```

4. Define correct directory to Rank-Calibration folder under 'file_path' and run it:
```bash
# For Linux/Mac
./run_compute_cma.sh

# For Windows
.\run_compute_cma.sh
```

5. Create table and visualization;
```bash
python create_table_visualization.py
```

## Acknowledgments

For original code and data, see [Rank calibration](https://github.com/shuoli90/Rank-Calibration)