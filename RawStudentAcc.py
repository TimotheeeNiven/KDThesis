import pandas as pd
import torch
import os

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv('/users/rniven1/GitHubRepos/RepDistiller/save/student_model/training_info.csv')
print("CSV file loaded successfully.")
print(df.head())  # Show the first few rows to confirm the data

# Path to the folder containing the model .pth files
model_base_path = "/users/rniven1/GitHubRepos/RepDistiller/save/models/"

# Function to extract accuracy from a .pth file
def extract_accuracy(model_path):
    print(f"Attempting to load model file: {model_path}")
    try:
        # Load the model data
        checkpoint = torch.load(model_path)
        print("Model loaded successfully.")
        
        # Assuming accuracy is saved in 'accuracy' or 'acc' in the checkpoint
        accuracy = checkpoint.get('accuracy', None) or checkpoint.get('acc', None)
        
        if accuracy is not None:
            # Convert to a CPU tensor if it's a CUDA tensor
            if isinstance(accuracy, torch.Tensor) and accuracy.is_cuda:
                accuracy = accuracy.cpu()
            print(f"Accuracy found: {accuracy}")
            return accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy
        else:
            print(f"Accuracy not found in checkpoint for {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model file {model_path}: {e}")
        return None

# Add a new column for RawStudentAccuracy
raw_accuracies = []

# Iterate over each row to find the student model and extract the accuracy
for idx, row in df.iterrows():
    student_model = row['Student Name']
    print(f"\nProcessing row {idx}: Student Model = {student_model}")
    
    # Construct path to the student's .pth file
    model_path = os.path.join(model_base_path, f"{student_model}_cifar100_lr_0.05_decay_0.0005_trial_0", "ckpt_epoch_240.pth")
    print(f"Constructed model path: {model_path}")
    
    # Extract accuracy
    accuracy = extract_accuracy(model_path)
    raw_accuracies.append(accuracy)
    print(f"Appended accuracy: {accuracy}")

# Add the raw accuracies to the dataframe
df['RawStudentAccuracy'] = raw_accuracies
print("\nUpdated DataFrame with RawStudentAccuracy column:")
print(df.head())  # Display the first few rows with the new column

# Save the updated DataFrame to a new CSV file
output_path = 'updated_file.csv'
df.to_csv(output_path, index=False)
print(f"\nCSV updated and saved to {output_path}")
