#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=pytorch_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00  # Increased time for better convergence
#SBATCH --mem=128G

export TORCH_SCRIPT=train_teacher.py
export MODEL=efficientnet_b7

# Training Hyperparameters
export BATCH_SIZE=32  # Increase to 128 or 256 if GPU allows
export LEARNING_RATE=0.01  # Lowered for stability
export LR_DECAY_EPOCHS="5,7,9"  # Adjusted for smoother decay
export LR_DECAY_RATE=0.1
export WEIGHT_DECAY=5e-4
export MOMENTUM=0.9
export EPOCHS=1  # Keep high for MobileNetV3
export DATASET=imagenet #speechcommands

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
conda activate IC  # Activate the Conda environment

python -c "import tensorboard_logger; print('tensorboard_logger is available')"

# List loaded modules (this may be empty, which is fine)
module list
echo ""
echo "Python: $(which python)"
echo "Executing: python $TORCH_SCRIPT"
echo ""

# Run training with optimized parameters
python $TORCH_SCRIPT --model $MODEL \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lr_decay_epochs $LR_DECAY_EPOCHS \
    --lr_decay_rate $LR_DECAY_RATE \
    --weight_decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --epochs $EPOCHS \
    --dataset $DATASET \

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
