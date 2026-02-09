import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

# Paths
DATA_PATH = "/mnt/ml-data/datasets"
MODEL_PATH = "/mnt/ml-data/models"
LOG_PATH = "/mnt/ml-data/logs"

# Target column
TARGET_COL = "income"

# Load datasets
processed_df = pd.read_csv(os.path.join(DATA_PATH, "processed.csv"))
raw_df = pd.read_csv(os.path.join(DATA_PATH, "raw.csv"), header=None, names=processed_df.columns)

# Combine datasets
df = pd.concat([raw_df, processed_df], ignore_index=True)

# Split features and target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X = pd.get_dummies(X, drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

best_model_name = None
best_accuracy = -1 
best_model = None

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.2f}")
    
    if acc >= best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model

# Save best model
model_file = os.path.join(MODEL_PATH, f"{best_model_name}_model.pkl")
joblib.dump(best_model, model_file)

# Log metrics
log_file = os.path.join(LOG_PATH, "metrics.log")
with open(log_file, "a") as f:
    f.write(f"Model: {best_model_name}\n")
    f.write(f"Accuracy: {best_accuracy:.2f}\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*30 + "\n")

print(f"Best model saved: {model_file}")
print(f"Metrics logged in: {log_file}")
print("Timestamp: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
