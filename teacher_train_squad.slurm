#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=squad_teacher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --mem=64G

# === Script and Model ===
export TORCH_SCRIPT=train_teacher.py
export MODEL=bertsmall

# === BERT Fine-tuning Hyperparameters ===
export BATCH_SIZE=32
export LEARNING_RATE=3e-5
export EPOCHS=10
export WEIGHT_DECAY=0.01
export DATASET=squad
export TOKENIZERS_PARALLELISM=false
export TOKENIZER=bert-base-uncased

# === Enhanced Features ===
export LR_SCHEDULER=cosine
export WARMUP_EPOCHS=5
export FREEZE_BACKBONE=--freeze_backbone

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

# === Load Conda environment ===
source /users/rniven1/miniforge3/etc/profile.d/conda.sh
conda activate IC

python -c "import tensorboard_logger; print('tensorboard_logger is available')"

module list
echo ""
echo "Python: $(which python)"
echo "Executing: python $TORCH_SCRIPT"
echo ""

# === Run BERT SQuAD training ===
python $TORCH_SCRIPT \
  --model $MODEL \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --epochs $EPOCHS \
  --dataset $DATASET \
  --optimizer adamw \
  --tokenizer $TOKENIZER \
  --lr_scheduler $LR_SCHEDULER \
  --warmup_epochs $WARMUP_EPOCHS \
  $FREEZE_BACKBONE

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
