#! /bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=pytorch_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=80:00:00
#SBATCH --mem=64G

export TORCH_SCRIPT=train_student_PnC.py
export MODEL=convnexttiny

export TRAINING_TYPE=kd
# Training Hyperparameters
export BATCH_SIZE=128
export LEARNING_RATE=0.00625  # Lowered for stability
export WEIGHT_DECAY=1e-5
export MOMENTUM=0.9
export EPOCHS=300
export DATASET=cifar100
export MIXUP=--mixup
export LR_SCHEDULER=cosine
export OPTIMIZER=adamw  # Use 'adamw' if supported
export RESIZE_INPUT=--resize_input  # Resizes to 256x256 input
export STRONG_AUG=--strong_aug
export LABEL_SMOOTHING=0.1
export GRAD_CLIP=1.0
export MASTER_PORT=$((10000 + RANDOM % 10000))

# Path to the teacher model (note the corrected closing quote)
export TEACHER_MODEL_PATH="/users/rniven1/GitHubRepos/RepDistiller/models/trained_teacher/convnextlarge_cifar100_lr_0.003_decay_0.05_trial_0/convnextlarge_best.pth"
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
conda activate test310_1  # Activate the Conda environment

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
    --optimizer $OPTIMIZER \
    --lr_scheduler $LR_SCHEDULER \
    --kd_T 2 \
    --gamma 0 \
    --alpha 1 \
    --beta 0 \
    --label_smoothing $LABEL_SMOOTHING \
    --grad_clip $GRAD_CLIP \
    $RESIZE_INPUT \
    $STRONG_AUG \
    $MIXUP


echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"

