#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=pytorch_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=80:00:00  # Increased time for better convergence
#SBATCH --mem=64G

export TORCH_SCRIPT=train_teacher.py
export MODEL=vit_pretrainedimagetiny

# === Training Hyperparameters for Transfer Learning ===
export BATCH_SIZE=128                  # Can go to 128 if memory allows
export LEARNING_RATE=0.0005           # Lower LR for pretrained backbone + mixup
export LR_DECAY_EPOCHS="150,180,210"  # Schedule decay milestones
export LR_DECAY_RATE=0.1
export WEIGHT_DECAY=0.05              # 0.05 works better for ConvNeXt
export MOMENTUM=0.9
export EPOCHS=10                    # Enough for full adaptation
export DATASET=tinyimagenet
export OPTIMIZER=adamw
export LR_SCHEDULER=cosine            # Smooth learning rate decay
export RESIZE_INPUT=--resize_input    # Resizes to 256x256 input
export WARMUP_EPOCHS=5                # Warmup helps stabilize ViT/ConvNeXt
export MASTER_PORT=$((10000 + RANDOM % 10000))

# === Overfitting Control ===
export MIXUP=--mixup
export LABEL_SMOOTHING="--label_smoothing 0.1"
export STRONG_AUG=--strong_aug

# === Print environment info ===
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

# === Load necessary Conda environment ===
source /users/rniven1/miniforge3/etc/profile.d/conda.sh
conda activate IC

python -c "import tensorboard_logger; print('tensorboard_logger is available')"

# === List loaded modules (optional) ===
module list
echo ""
echo "Python: $(which python)"
echo "Executing: python $TORCH_SCRIPT"
echo ""

# === Run training with all flags ===
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TORCH_SCRIPT \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lr_decay_epochs $LR_DECAY_EPOCHS \
    --lr_decay_rate $LR_DECAY_RATE \
    --weight_decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --optimizer $OPTIMIZER \
    --epochs $EPOCHS \
    --dataset $DATASET \
    --lr_scheduler $LR_SCHEDULER \
    --warmup_epochs $WARMUP_EPOCHS \
    $RESIZE_INPUT \
    $MIXUP \
    $LABEL_SMOOTHING \
    $STRONG_AUG

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
