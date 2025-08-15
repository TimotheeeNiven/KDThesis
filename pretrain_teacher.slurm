#! /bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=convert_effnet_model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --exclude=str-gpu21

# Path to the script that loads, processes, and saves the pretrained EfficientNet model
export CONVERT_SCRIPT=convert_effnet_model.py  # Name of the script that performs the conversion

# Path to the pretrained EfficientNet model (.pth file)
export PRETRAINED_MODEL_PATH="/users/rniven1/GitHubRepos/RepDistiller/models/pretrained_teachers/efficientnet_b2_imagenet/efficientnet_b2.pth"  # Change this to your pretrained model path

# Path where the model will be saved after conversion
export SAVED_MODEL_PATH="/users/rniven1/GitHubRepos/RepDistiller/models/trained_teacher/efficientnet_b2_imagenet/efficientnet_b2.pth"  # Change this to where you want the converted model saved

# Model type (EfficientNet variant)
export MODEL_TYPE="efficientnet_b2"  # Example: Change this to the desired EfficientNet variant (e.g., efficientnet_b1, efficientnet_b2, etc.)

# Environment setup
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

# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

echo ""
echo "Python: $(which python)"  # Verify which Python is being used
echo "Executing: python $CONVERT_SCRIPT"
echo ""

# Run the model conversion script to load pretrained model, process, and save it
python $CONVERT_SCRIPT --model $MODEL_TYPE \
    --pretrained_model_path $PRETRAINED_MODEL_PATH \
    --saved_model_path $SAVED_MODEL_PATH

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
