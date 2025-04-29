import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE


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


def train_decision_tree_model(X_train, y_train, max_depth=None, random_state=42):
    # Initialize and train the Decision Tree model
    dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_model.fit(X_train, y_train)
    return dt_model


def evaluate_model(model, X_test, y_test, label_encoder):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Decode numeric labels to original string labels
    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(y_pred)

    print("Classification Report:\n", classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, f1


# Paths to your CSV files
train_path = 'C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/mainwork/Train_Test_Dataset/Train_Dataset.csv'
test_path = 'C:/Users/Admin/Downloads/CSVs-20240207T040926Z-001/CSVs/mainwork/Train_Test_Dataset/Test_Dataset.csv'

# Load and preprocess the data
X_train, X_test, y_train, y_test, label_encoder = load_data(train_path, test_path)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to balance the training data (uncomment if needed)
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# Train and evaluate the model
dt_model = train_decision_tree_model(X_train, y_train, max_depth=None, random_state=42)
accuracy, precision, f1 = evaluate_model(dt_model, X_test, y_test, label_encoder)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the trained Decision Tree model
model_path = "decision_tree_model.pkl"
joblib.dump(dt_model, model_path)
print(f"Model saved to {model_path}")
