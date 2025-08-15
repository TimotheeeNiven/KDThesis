import os
import shutil

VAL_DIR = "/users/rniven1/GitHubRepos/RepDistiller/data/imagenet/val"
GT_FILE = "/users/rniven1/GitHubRepos/RepDistiller/data/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"

# Read ground truth labels
with open(GT_FILE, "r") as f:
    labels = [int(line.strip()) for line in f.readlines()]

assert len(labels) == 50000, "Ground truth file should contain 50,000 entries"

# Create folders for each class
for i in range(1, 1001):
    class_dir = os.path.join(VAL_DIR, f"{i:04d}")
    os.makedirs(class_dir, exist_ok=True)

# Move each image into the correct class folder
for idx, label in enumerate(labels, start=1):
    img_name = f"ILSVRC2012_val_{idx:08d}.JPEG"
    src = os.path.join(VAL_DIR, img_name)
    dst = os.path.join(VAL_DIR, f"{label:04d}", img_name)
    if os.path.exists(src):
        shutil.move(src, dst)

print("? Validation set successfully organized into class folders.")
