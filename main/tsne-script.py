import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Configuration
input_csv = "C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/encoded.csv"
output_csv = "C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/pca_transformed_data-tsne.csv"
save_dir = "C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/tsne"
n_components = 10
batch_size = 10000
label_col = "Label"
tsne_sample_size = 5000  # Number of samples to use for t-SNE (for performance)

# 1. Initialize objects
scaler = StandardScaler()
pca = IncrementalPCA(n_components=n_components)

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
full_labels = []
full_pca_features = []

for chunk in pd.read_csv(input_csv, chunksize=batch_size):
    # Get features and labels separately
    labels = chunk[label_col]
    features = chunk.drop(columns=[label_col], errors='ignore')
    
    # Transform features
    scaled_data = scaler.transform(features)
    transformed_data = pca.transform(scaled_data)
    
    # Store for t-SNE visualization
    full_labels.append(labels.values)
    full_pca_features.append(transformed_data)
    
    # Create DataFrame with PC columns and add Label
    pc_df = pd.DataFrame(transformed_data, 
                        columns=[f"PC_{i+1}" for i in range(n_components)])
    pc_df[label_col] = labels.values
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


# 5. t-SNE Validation Visualization
print("\nPreparing t-SNE validation visualizations...")

# Combine all data for sampling
all_labels = np.concatenate(full_labels)
all_pca_features = np.vstack(full_pca_features)

# Convert string labels to numeric values for coloring
unique_labels = np.unique(all_labels)
label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
numeric_labels = np.array([label_to_num[label] for label in all_labels])

# Sample data for t-SNE (for performance)
if len(all_labels) > tsne_sample_size:
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(all_labels), tsne_sample_size, replace=False)
    sample_features = all_pca_features[sample_idx]
    sample_numeric_labels = numeric_labels[sample_idx]
    sample_string_labels = all_labels[sample_idx]
else:
    sample_features = all_pca_features
    sample_numeric_labels = numeric_labels
    sample_string_labels = all_labels

# Run t-SNE on PCA-reduced features
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_features = tsne.fit_transform(sample_features)

# Plot t-SNE results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                     c=sample_numeric_labels, alpha=0.6, cmap='viridis')

# Create legend with actual label names
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor=plt.cm.viridis(i/len(unique_labels)), 
                markersize=10, label=label) 
               for i, label in enumerate(unique_labels)]
plt.legend(handles=legend_labels, title='Classes')

plt.title(f"t-SNE Visualization of PCA-Reduced Features (n={len(sample_string_labels)})")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# Save the plot
tsne_plot_path = f"{save_dir}/pca_tsne_validation.png"
plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved t-SNE validation plot to {tsne_plot_path}")

# Optional: Compare with t-SNE on original scaled data (for reference)
print("Running t-SNE on original scaled data for comparison...")
# Load a sample of original data
original_sample = pd.read_csv(input_csv, nrows=tsne_sample_size)
original_string_labels = original_sample[label_col]
original_numeric_labels = np.array([label_to_num[label] for label in original_string_labels])
original_features = original_sample.drop(columns=[label_col], errors='ignore')
scaled_original = scaler.transform(original_features)

tsne_original = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_original_features = tsne_original.fit_transform(scaled_original)

# Plot comparison
plt.figure(figsize=(12, 8))
scatter = plt.scatter(tsne_original_features[:, 0], tsne_original_features[:, 1], 
                     c=original_numeric_labels, alpha=0.6, cmap='viridis')

# Create legend
plt.legend(handles=legend_labels, title='Classes')
plt.title(f"t-SNE Visualization of Original Scaled Data (n={len(original_string_labels)})")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# Save the comparison plot
tsne_original_plot_path = f"{save_dir}/original_tsne_comparison.png"
plt.savefig(tsne_original_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved original data t-SNE plot to {tsne_original_plot_path}")

print("\nValidation complete! Compare the two plots:")
print(f"1. PCA + t-SNE: {tsne_plot_path}")
print(f"2. Original data t-SNE: {tsne_original_plot_path}")
print("\nIf clusters are similarly separated in both, PCA preserved the structure well.")