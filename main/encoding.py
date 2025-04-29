import pandas as pd
from category_encoders import TargetEncoder
from datetime import datetime
import pickle
import os

def parse_creation_date_time(date_str):
    """Convert date string to Unix timestamp (seconds). Returns 0 if invalid."""
    if pd.isna(date_str) or str(date_str).strip() in ("0", "0000-00-00 00:00:00"):
        return 0
    formats = [
        "%d-%m-%Y %I.%M.%S %p",  # e.g., "15-09-1997 4.00.00 AM"
        "%m-%d-%Y %I.%M.%S %p",  # e.g., "09-15-1997 4.00.00 AM"
        "%Y-%m-%d %H:%M:%S",      # e.g., "1997-09-15 04:00:00"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(str(date_str).strip(), fmt)
            return int(dt.timestamp())
        except ValueError:
            continue
    return 0  # Fallback

def target_encode_csv(input_csv, output_csv, target_column, encoder_save_path):
    # Load data
    df = pd.read_csv(input_csv)
    
    # Convert Creation_Date_Time
    df["Creation_Date_Time"] = df["Creation_Date_Time"].apply(parse_creation_date_time)
    
    # Target encode categorical columns
    categorical_cols = ["tld", "sld","longest_meaningful_word", "Domain_Name", "Registrar", "Registrant_Name", "Emails", "Organization", "State", "Country.1"]
    encoder = TargetEncoder(cols=categorical_cols, smoothing=10.0)
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df[target_column])
    
    # Save encoded data and encoder
    df.to_csv(output_csv, index=False)
    with open(encoder_save_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Encoded data saved to {output_csv}")
    print(f"Encoder saved to {encoder_save_path}")

# ===== CONFIGURATION =====
input_csv = "C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/additional_feats_final.csv"   # Input CSV path
output_csv = "C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/work12/encoded.csv"     # Output CSV path
target_column = "Label"     # Your target column
encoder_save_path = "target_encoder1.pkl" # Path to save the encoder
# =========================

if __name__ == "__main__":
    target_encode_csv(input_csv, output_csv, target_column, encoder_save_path)



