import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "pose_features1.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model1.pkl")

# Load the landmark features
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded dataset with shape: {df.shape}")

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train/test

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train a Random Forest Classifier
print("[INFO] Training model...")
clf = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n[RESULT] Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("[INFO] Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save the model and label encoder
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump({"model": clf, "label_encoder": le}, MODEL_PATH)
print(f"[SUCCESS] Model saved at: {MODEL_PATH}")
