from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from feature_extraction import extract_features
import pickle
from check import check_domain
from white_check import in_whitelist

# Load the trained model
MODEL_PATH = "knn_model_all.pkl"
model = joblib.load(MODEL_PATH)

# Initialize FastAPI
app = FastAPI()


# PCA transformation
def apply_pca(df):
    # pca = PCA(n_components=n_components)
    # return pd.DataFrame(pca.fit_transform(features))
    with open(f"scaler_pca.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"pca.pkl", "rb") as f:
        pca = pickle.load(f)

    # Ensure the input is 2D array (single row)
    if len(df.shape) == 1:
        df = df.values.reshape(1, -1)
    else:
        df = df.values

    scaled_data = scaler.transform(df)
    transformed_data = pca.transform(scaled_data)

    # print("transformed data pca")
    # print(pd.DataFrame(transformed_data, columns=[f"PC_{i+1}" for i in range(transformed_data.shape[1])]))

    return pd.DataFrame(transformed_data, columns=[f"PC_{i+1}" for i in range(transformed_data.shape[1])])


def preprocess_single_domain(domain_data, encoder_path):
    # Load the saved encoder
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    # Convert to DataFrame (single row)
    df = pd.DataFrame([domain_data])
    
    # Apply encoding (assuming 'Creation_Date_Time' is already in Unix time)
    categorical_cols = ["tld", "sld","longest_meaningful_word", "Domain_Name", "Registrar", "Registrant_Name", "Emails", "Organization", "State", "Country.1"]
    df[categorical_cols] = encoder.transform(df[categorical_cols])
    
    return df

@app.post("/predict/")
async def predict_domain(data: dict):
    domain = data.get("domain")

    # Check if the domain exists in the whitelist
    if(in_whitelist(domain)):
        return {
            "domain": domain,
            "isUntrusted": False
        }

    # Check if the domain exists in the blacklist
    if(check_domain(domain)):
        return {
            "domain": domain,
            "isUntrusted": True
        }
    
    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required")

    # Extract features
    features = extract_features(domain)

    if not features:
        raise HTTPException(status_code=500, detail="Feature extraction failed")

    # Preprocess features
    encoded_features = preprocess_single_domain(features, "target_encoder1.pkl")

    # Apply PCA transformation
    pca_features = apply_pca(encoded_features)

    prediction = model.predict(pca_features)
    # print(prediction)

    return {
        "domain": domain,
        "isUntrusted": bool(prediction != 0)
    }

