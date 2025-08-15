import csv

# Create CSV file and write the header
def create_csv_file(filename):
    headers = ['epoch', 'student_accuracy_with_kd', 'student_loss_with_kd', 
               'student_accuracy_without_kd', 'student_loss_without_kd']
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"CSV file '{filename}' created successfully.")

# Call the function to create the CSV
create_csv_file('training_results.csv')
