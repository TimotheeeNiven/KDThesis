#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=pytorch_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --nodelist=str-gpu21  # Request the A100 node

export TORCH_SCRIPT=train_teacher.py
export MODEL=wrn_40_1

# Print environment info
GPU_MDL=$(grep ^Model: /proc/driver/nvidia/gpus/*/information | tr '\t' ' ' | head -1)
export GPU_MDL=${GPU_MDL##* }
export NUM_GPUS=$(echo ${SLURM_JOB_GPUS} | tr ',' ' ' | wc -w)
export PYTHONUNBUFFERED=TRUE

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Node List   : $SLURM_JOB_NODELIST"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "Num GPUs    : $NUM_GPUS $GPU_MDL"
echo "======================================================"
echo ""

# Load necessary Conda environment
source /users/rniven1/miniforge3/etc/profile.d/conda.sh  # Ensure this is the correct path
conda activate myenv  # Activate the Conda environment

# Check TensorBoard logger availability
python -c "import tensorboard_logger; print('tensorboard_logger is available')"

# List loaded modules (this may be empty, which is fine)
module list
echo ""
echo "Python: $(which python)"
echo "Executing: python $TORCH_SCRIPT"
echo ""
python $TORCH_SCRIPT --model $MODEL  # Run the training script

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
