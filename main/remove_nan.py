import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/All_categories.csv')

# Replace NaN values with a specified value, for example 0
df.fillna(0, inplace=True)

# Alternatively, to replace NaNs with the mean of each column:
# df.fillna(df.mean(), inplace=True)

# Save the updated DataFrame back to a CSV file
df.to_csv('C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/All_categories_nan_removed.csv', index=False)

print("NaN values replaced and file saved as 'your_file_filled.csv'")
