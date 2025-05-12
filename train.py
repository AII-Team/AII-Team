# train.py - Customer Purchase Intent Prediction
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("Deep_clean.csv")

# ===== CONFIGURATION =====
TARGET_COLUMN = "Purchase_Intent"  # CHANGE THIS to your target column
ID_COLUMN = "Customer_ID"          # Column to remove (not a feature)

# Preprocessing
def preprocess(df):
    df = df.drop(ID_COLUMN, axis=1)  # Remove ID column
    
    # Convert strings to numbers (e.g., "Male"/"Female" → 0/1)
    for col in df.select_dtypes(include=['object']).columns:
        if col != TARGET_COLUMN:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

data_clean = preprocess(data.copy())

# Prepare features/target
X = data_clean.drop(TARGET_COLUMN, axis=1)
y = data_clean[TARGET_COLUMN]

# If target is text (e.g., "Yes"/"No"), convert to numbers
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Tracking
mlflow.set_experiment("Customer_Purchase_Prediction")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Log details
    mlflow.log_params({"n_estimators": 100, "features": X.shape[1]})
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model trained! Accuracy: {accuracy:.2f}")
    print("Run 'mlflow ui' to view results.")