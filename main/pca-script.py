import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# Configuration
input_csv = "C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/encoded.csv"  # Replace with your file path
output_csv = "C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/pca_transformed_data.csv"  # Output CSV with PCA features
save_dir = "C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12"
n_components = 10  # Set to None for automatic selection
batch_size = 10000  # Process data in chunks
label_col = "Label"  # Column to exclude from PCA but include in output

# 1. Initialize objects
scaler = StandardScaler()
pca = IncrementalPCA(n_components=n_components)

# 2. Fit Scaler and PCA (excluding Label column)
# 2. Fit Scaler and PCA (excluding Label column)
print("Fitting models...")
for chunk in pd.read_csv(input_csv, chunksize=batch_size):
    features = chunk.drop(columns=[label_col], errors='ignore')
    scaler.partial_fit(features)
    
for chunk in pd.read_csv(input_csv, chunksize=batch_size):
    features = chunk.drop(columns=[label_col], errors='ignore')
    scaled_data = scaler.transform(features)
    pca.partial_fit(scaled_data)

# 3. Transform data and save with Label column
print("Transforming data and saving...")
transformed_chunks = []

for chunk in pd.read_csv(input_csv, chunksize=batch_size):
    # Get features and labels separately
    labels = chunk[label_col]
    features = chunk.drop(columns=[label_col], errors='ignore')
    
    # Transform features
    scaled_data = scaler.transform(features)
    transformed_data = pca.transform(scaled_data)
    
    # Create DataFrame with PC columns and add Label
    pc_df = pd.DataFrame(transformed_data, 
                        columns=[f"PC_{i+1}" for i in range(n_components)])
    pc_df[label_col] = labels.values  # Now using 1D array
    transformed_chunks.append(pc_df)

# Combine and save
pd.concat(transformed_chunks).to_csv(output_csv, index=False)
print(f"Saved PCA-transformed data with labels to {output_csv}")

# 4. Save models
print("Saving models...")
Path(save_dir).mkdir(exist_ok=True)
with open(f"{save_dir}/scaler_pca.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{save_dir}/pca.pkl", "wb") as f:
    pickle.dump(pca, f)


