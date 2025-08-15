#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=resnet110
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --time=120:00:00
#SBATCH --mem=64G

export TORCH_SCRIPT=train_teacher.py
export MODEL=MobileNetV2

# === Training Hyperparameters ===
export BATCH_SIZE=128
export LEARNING_RATE=0.05
export LR_DECAY_EPOCHS="150,180,210"
export LR_DECAY_RATE=0.1
export WEIGHT_DECAY=5e-4
export EPOCHS=300
export DATASET=tinyimagenet
export LR_SCHEDULER=step
export OPTIMIZER=sgd  # Use 'adamw' if supported
export MASTER_PORT=$((10000 + RANDOM % 10000))

# === Environment Info ===
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

# === Load environment ===
source /users/rniven1/miniforge3/etc/profile.d/conda.sh
conda activate IC

python -c "import torch; print(torch.cuda.is_available())"

echo ""
echo "Python: $(which python)"
echo "Executing: python $TORCH_SCRIPT"
echo ""

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TORCH_SCRIPT \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --lr_scheduler $LR_SCHEDULER \
    --lr_decay_epochs $LR_DECAY_EPOCHS \
    --lr_decay_rate $LR_DECAY_RATE \
    --dataset $DATASET 
echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
