#! /bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=pytorch_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --time=16:00:00
#SBATCH --mem=64G

export TORCH_SCRIPT=train_student.py
export MODEL=MobileNetV2

export TRAINING_TYPE=attention
# Training Hyperparameters
export BATCH_SIZE=128  # 64
export LEARNING_RATE=0.05  # Lowered for stability
export WEIGHT_DECAY=5e-4
export MOMENTUM=0.9
export EPOCHS=300
export DATASET=tinyimagenet
export MASTER_PORT=$((10000 + RANDOM % 10000))

# Path to the teacher model (note the corrected closing quote)
export TEACHER_MODEL_PATH="/users/rniven1/GitHubRepos/RepDistiller/models/trained_teacher/wrn_40_2_tinyimagenet_lr_0.05_decay_0.0005_trial_0/wrn_40_2_best.pth"
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

export TOKENIZERS_PARALLELISM=false

python -c "import torch; print(torch.cuda.is_available())"

echo ""
echo "Python: $(which python)"  # Verify which Python is being used
echo "Executing: python $TORCH_SCRIPT"
echo ""

# Run your script with the student model, teacher model path, alpha, beta, and cross-loss
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TORCH_SCRIPT \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --path_t $TEACHER_MODEL_PATH \
    --distill $TRAINING_TYPE \
    --dataset $DATASET \
    --gamma 1.0 \
    --alpha 0.1 \
    --beta 0.9

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
