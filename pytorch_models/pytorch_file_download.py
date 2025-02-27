from transformers import AutoImageProcessor, MobileNetV1ForImageClassification
import torch

# Step 1: Load the pre-trained MobileNetV1 model and processor
print("Downloading the MobileNetV1 model and processor...")
processor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_0.75_192")
model = MobileNetV1ForImageClassification.from_pretrained("google/mobilenet_v1_0.75_192")

# Step 2: Save the model as a .pth file
output_path = "./mobilenet_v1_0.75_192_model.pth"
print(f"Saving the model to: {output_path}...")

# Save only the model state dict (this reduces file size compared to saving the entire model object)
torch.save(model.state_dict(), output_path)

print(f"Model saved successfully to: {output_path}")

# Step 3 (Optional): Verify the saved .pth file by loading it
print("Verifying the saved .pth file...")
loaded_model = MobileNetV1ForImageClassification.from_pretrained(
    "google/mobilenet_v1_0.75_192"
)
loaded_model.load_state_dict(torch.load(output_path))
print("Model loaded successfully from the .pth file!")
