import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib

def load_data(train_path, test_path, label_column='Label'):
    # Load the train and test datasets
    train_data = pd.read_csv(train_path, low_memory=False)
    test_data = pd.read_csv(test_path, low_memory=False)

    # Separate features and labels
    X_train = train_data.drop(columns=[label_column])
    y_train = train_data[label_column]
    X_test = test_data.drop(columns=[label_column])
    y_test = test_data[label_column]

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)  # Fit and transform on training labels
    y_test = label_encoder.transform(y_test)       # Transform on test labels

    return X_train, X_test, y_train, y_test, label_encoder

def train_xgboost_model(X_train, y_train, learning_rate=0.1, max_depth=6, n_estimators=100):
    # Initialize the XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',  # Use 'binary:logistic' for binary classification
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        use_label_encoder=False,  # Disable warning for LabelEncoder usage
        eval_metric='mlogloss'   # Use 'logloss' for binary classification
    )
    
    # Train the XGBoost model
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test, label_encoder):
    # Make predictions
    y_pred = model.predict(X_test)

    # Decode numeric labels back to original string labels
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print("Classification Report:\n", classification_report(y_test_decoded, y_pred_decoded))
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1

# Paths to your CSV files
train_path = 'C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/mainwork/Train_Test_Dataset/Train_Dataset.csv'
test_path = 'C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/mainwork/Train_Test_Dataset/Test_Dataset.csv'

# Load and preprocess the data
X_train, X_test, y_train, y_test, label_encoder = load_data(train_path, test_path)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the XGBoost model
xgb_model = train_xgboost_model(X_train, y_train, learning_rate=0.1, max_depth=6, n_estimators=100)

# Evaluate the model
accuracy, precision, recall, f1 = evaluate_model(xgb_model, X_test, y_test, label_encoder)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the trained XGBoost model
model_path = "xgboost_model.pkl"
joblib.dump(xgb_model, model_path)
print(f"Model saved to {model_path}")
