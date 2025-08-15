# KDThesis – Built on RepDistiller

This repository is the working codebase for my Knowledge Distillation thesis.  
It is based on and adapted from the original **RepDistiller** repository by Yonglong Tian et al.

Original RepDistiller repository: [https://github.com/HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller)  
Original paper: *Contrastive Representation Distillation* (ICLR 2020) — [arXiv:1910.10699](http://arxiv.org/abs/1910.10699)

---

## Installation

This repository uses two separate requirements files:

- **`requirements_py.txt`** ? contains CUDA 11.8–specific PyTorch/TensorFlow GPU packages.
- **`requirements.txt`** ? contains all other Python dependencies.

### 1. Create a new conda environment
```bash
conda create -n kdthesis python=3.10 -y
conda activate kdthesis
```
### 2. Install CUDA 11.8–specific packages
Use the PyTorch wheel index to install the GPU-specific dependencies:
```bash
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu118 -r requirements_py.txt
```
### 3. Install the remaining dependencies
```bash
pip install -r requirements.txt
```
### 4. Verify installation
Run the following to confirm both PyTorch and TensorFlow are installed correctly and that CUDA is available:
```bash
python - <<'PY'
import torch, tensorflow as tf
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("TensorFlow:", tf.__version__)
PY
```
## Slurm server specific information

Certain SLURM files are set up in this repo to run different jobs. If you are running on a SLURM server, building upon these files is recommended.

The main driver scripts are `train_teacher.py` and `train_student.py`; everything else is a derivative of these files to help with workflow.

`PnC.slurm` and `train_student_PnC.py` are attempts at implementing the ideas from the *Patient and Consistent* paper: [Knowledge Distillation: A Good Teacher Is Patient and Consistent](https://arxiv.org/abs/2106.05237).

All the other SLURM files are set up for different instances, usually explained in the filenames.
