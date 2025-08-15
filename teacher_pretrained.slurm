#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=save_teacher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G

# === Environment Setup ===
source /users/rniven1/miniforge3/etc/profile.d/conda.sh
conda activate IC

# === Print Info ===
echo "======================================================"
echo "Start Time  : $(date)"
echo "Node List   : $SLURM_JOB_NODELIST"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "======================================================"
echo ""

# === Run Script ===
echo "Running: save_pretrained_teacher.py"
python teacher_pretrained.py

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
