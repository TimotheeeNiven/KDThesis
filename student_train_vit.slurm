#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=vit_small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --time=100:00:00
#SBATCH --mem=64G

export TORCH_SCRIPT=train_student.py
export MODEL=vit_tiny             # ? Match your model_dict exactly
export TRAINING_TYPE=attention

# === Distillation Hyperparameters ===
export ALPHA=0.0
export BETA=1.0
export CROSSLOSS=1.0
export KD_TEMP=4.0

# === Training Hyperparameters ===
export BATCH_SIZE=128
export LEARNING_RATE=0.001
export WEIGHT_DECAY=0.05
export EPOCHS=300
export WARMUP_EPOCHS=20
export DATASET=tinyimagenet
export OPTIMIZER=adamw  # Use 'adamw' if supported
export RESIZE_INPUT=--resize_input  # Resizes to 256x256 input
export STRONG_AUG=--strong_aug
export LABEL_SMOOTHING=0.1
export GRAD_CLIP=1.0
export MIXUP=--mixup  # optionally switch to --cutmix
export MASTER_PORT=$((10000 + RANDOM % 10000))


# === Teacher Model Path ===
export TEACHER_MODEL_PATH="/users/rniven1/GitHubRepos/RepDistiller/models/trained_teacher/vit_pretrainedimagetiny_tinyimagenet_lr_0.0005_decay_0.05_trial_0/vit_pretrainedimagetiny_best.pth"

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
    --warmup_epochs $WARMUP_EPOCHS \
    --optimizer $OPTIMIZER \
    --dataset $DATASET \
    --path_t $TEACHER_MODEL_PATH \
    --alpha $ALPHA \
    --beta $BETA \
    --gamma $CROSSLOSS \
    --distill $TRAINING_TYPE \
    --kd_T $KD_TEMP \
    --label_smoothing $LABEL_SMOOTHING \
    --grad_clip $GRAD_CLIP \
    $RESIZE_INPUT \
    $STRONG_AUG \
    $MIXUP \


echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
