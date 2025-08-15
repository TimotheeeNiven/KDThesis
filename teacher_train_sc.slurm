#! /bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=pytorch_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --exclude=str-gpu21

export TORCH_SCRIPT=train_student.py
export MODEL=MobileNetV2 # Specify the student model

export TRAINING_TYPE=attention

# KD Hyperparameters
export ALPHA=0.9  # Modify as necessary
export BETA=0   # Modify as necessary
export CROSSLOSS=0.1

# Training Hyperparameters
export BATCH_SIZE=256  # 64
export LEARNING_RATE=0.05  # Lowered for stability
export LR_DECAY_EPOCHS="10,15"  # Adjusted for smoother decay
export LR_DECAY_RATE=0.1
export WEIGHT_DECAY=5e-4
export MOMENTUM=0.9
export EPOCHS=20  # Keep high for MobileNetV3
export DATASET=speechcommands #speechcommands cifar100

# Path to the teacher model (note the corrected closing quote)
export TEACHER_MODEL_PATH="/users/rniven1/GitHubRepos/RepDistiller/models/trained_teacher_sc/MobileNetV2_speechcommands_lr_0.005_decay_0.0001_trial_0/MobileNetV2_best.pth"
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


python -c "import torch; print(torch.cuda.is_available())"

echo ""
echo "Python: $(which python)"  # Verify which Python is being used
echo "Executing: python $TORCH_SCRIPT"
echo ""

# Run your script with the student model, teacher model path, alpha, beta, and cross-loss
python $TORCH_SCRIPT --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lr_decay_epochs $LR_DECAY_EPOCHS \
    --lr_decay_rate $LR_DECAY_RATE \
    --weight_decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --path_t $TEACHER_MODEL_PATH \
    --alpha $ALPHA \
    --beta $BETA \
    --gamma $CROSSLOSS \
    --distill $TRAINING_TYPE \
    --dataset $DATASET \

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
