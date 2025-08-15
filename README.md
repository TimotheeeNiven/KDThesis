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
