#!/bin/bash
#SBATCH --job-name=stat_calc
#SBATCH --error=stat_processing_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --array=0-99  # Process 100 chunks, adjust as needed

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Load necessary modules (adjust as needed for your cluster)
source $HOME/miniconda3/conda.sh
conda activate precip_env

# Process the chunk corresponding to this array job's index
python split_process_chunks.py $SLURM_ARRAY_TASK_ID