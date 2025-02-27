#! /bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=pytorch_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --mem=64G

export TORCH_SCRIPT=train_student.py
export MODEL=wrn_16_2  # Specify the student model

export TRAINING_TYPE=kd

export ALPHA=0.9  # Modify as necessary
export BETA=0   # Modify as necessary
export CROSSLOSS=0.1
export EPOCH=240

# Path to the teacher model (note the corrected closing quote)
export TEACHER_MODEL_PATH="/users/rniven1/GitHubRepos/RepDistiller/models/trained_teacher/wide_resnet101_2/wide_resnet101_2_cifar_distilled.pth"

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


python -c "import torch; print(torch.cuda.is_available())"

echo ""
echo "Python: $(which python)"  # Verify which Python is being used
echo "Executing: python $TORCH_SCRIPT"
echo ""

# Run your script with the student model, teacher model path, alpha, beta, and cross-loss
python $TORCH_SCRIPT --model $MODEL --epochs $EPOCH --path_t $TEACHER_MODEL_PATH  --alpha $ALPHA --beta $BETA --gamma $CROSSLOSS --distill $TRAINING_TYPE

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
