import tensorflow as tf
import torch
import os

def keras_to_pytorch(keras_model_path, output_pth_path):
    """
    Converts a Keras model (.keras) to PyTorch weights (.pth).

    Args:
        keras_model_path (str): Path to the Keras model file (.keras).
        output_pth_path (str): Path to save the PyTorch weights file (.pth).
    """
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model file not found: {keras_model_path}")

    print(f"Loading Keras model from: {keras_model_path}")
    keras_model = tf.keras.models.load_model(keras_model_path)
    keras_weights = keras_model.get_weights()

    # Convert Keras weights to PyTorch format
    pytorch_weights = {}
    print("Converting weights...")

    for i, weight in enumerate(keras_weights):
        pytorch_weights[f"layer_{i}"] = torch.tensor(weight)

    print(f"Saving PyTorch weights to: {output_pth_path}")
    torch.save(pytorch_weights, output_pth_path)
    print("Conversion complete.")

# Example usage
if __name__ == "__main__":
    # File paths
    keras_file = "cifar100_effnet_ft.keras"  # Path to the .keras file
    pytorch_file = "cifar100_effnet_ft.pth"  # Path to save the .pth file

    keras_to_pytorch(keras_file, pytorch_file)
