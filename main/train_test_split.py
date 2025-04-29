import os
import pandas as pd
from sklearn.model_selection import train_test_split

def combine_csv_files(input_directory, output_directory, input_csv_filename, label_column, train_filename="Train_Dataset.csv", test_filename="Test_Dataset.csv", train_size=0.9):
    # Define paths for input and output
    input_csv_path = os.path.join(input_directory, input_csv_filename)
    
    # Read the input CSV file
    df = pd.read_csv(input_csv_path)
    
    # Check if the label column exists
    if label_column not in df.columns:
        raise ValueError(f"The specified label column '{label_column}' does not exist in the dataset.")
    
    # Ensure all unique labels are included in the training set
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=42,
        stratify=df[label_column]
    )
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Save the training and test datasets to the output directory
    train_output_path = os.path.join(output_directory, train_filename)
    test_output_path = os.path.join(output_directory, test_filename)
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"Train dataset saved as: {train_output_path}")
    print(f"Test dataset saved as: {test_output_path}")

# Example usage
# Specify the label column (e.g., 'Category' or 'Label')
input_directory = 'C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12'
output_directory = 'C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/Train_Test_Dataset'
input_csv_filename = 'pca_transformed_data.csv'
label_column = 'Label'  # Update with the name of your label column

combine_csv_files(input_directory, output_directory, input_csv_filename, label_column)
