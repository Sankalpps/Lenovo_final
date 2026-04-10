"""
Train XGBoost Model for NVMe Failure Prediction
================================================
Loads the dataset, trains the model, and saves it with compatible versions
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Paths
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "Frontend")
DATA_PATH = os.path.join(FRONTEND_DIR, "data", "NVMe_Drive_Failure_Dataset.csv")
MODEL_DIR = os.path.join(FRONTEND_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Features
BASE_FEATURES = [
    "Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB",
    "Temperature_C", "Percent_Life_Used", "Media_Errors",
    "Unsafe_Shutdowns", "CRC_Errors", "Read_Error_Rate",
    "Write_Error_Rate"
]

FAILURE_MODE_LABELS = {
    0: "No Failure",
    1: "Wear-Out Failure",
    2: "Thermal Failure",
    3: "Firmware Failure",
    4: "Media Error Failure",
    5: "Unsafe Shutdown Failure",
}

def train_model():
    print("="*80)
    print("TRAINING XGBOOST MODEL FOR NVME FAILURE PREDICTION")
    print("="*80)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return False
    
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = BASE_FEATURES + ["Failure_Mode"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ERROR: Missing columns: {missing_cols}")
        print(f"  Available: {list(df.columns)}")
        return False
    
    # Prepare data
    print("\n[2/5] Preparing data...")
    X = df[BASE_FEATURES].values
    y = df["Failure_Mode"].values
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Failure mode distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for mode, count in zip(unique, counts):
        label = FAILURE_MODE_LABELS.get(mode, f"Unknown {mode}")
        print(f"    - {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split data
    print("\n[3/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Preprocess
    print("\n[4/5] Preprocessing...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"  Scaler fitted")
    print(f"  Label encoder fitted: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Train model
    print("\n[5/5] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0,
        tree_method='hist'
    )
    
    model.fit(
        X_train_scaled, y_train_encoded,
        eval_set=[(X_test_scaled, y_test_encoded)],
        verbose=False
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train_encoded, y_pred_train)
    test_acc = accuracy_score(y_test_encoded, y_pred_test)
    
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc*100:.2f}%")
    print(f"  Test:  {test_acc*100:.2f}%")
    
    print(f"\nPrecision: {precision_score(y_test_encoded, y_pred_test, average='weighted')*100:.2f}%")
    print(f"Recall:    {recall_score(y_test_encoded, y_pred_test, average='weighted')*100:.2f}%")
    print(f"F1-Score:  {f1_score(y_test_encoded, y_pred_test, average='weighted')*100:.2f}%")
    
    print(f"\nClassification Report:")
    report = classification_report(
        y_test_encoded, y_pred_test,
        target_names=[FAILURE_MODE_LABELS.get(mode, f"Mode {mode}") for mode in label_encoder.classes_],
        digits=4
    )
    print(report)
    
    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    try:
        joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
        print(f"  Saved: xgb_model.pkl")
        
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        print(f"  Saved: scaler.pkl")
        
        joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
        print(f"  Saved: label_encoder.pkl")
        
        # Save metadata
        metadata = {
            "accuracy": float(test_acc),
            "precision": float(precision_score(y_test_encoded, y_pred_test, average='weighted')),
            "recall": float(recall_score(y_test_encoded, y_pred_test, average='weighted')),
            "f1_score": float(f1_score(y_test_encoded, y_pred_test, average='weighted')),
            "failure_modes": FAILURE_MODE_LABELS,
            "features": BASE_FEATURES,
            "num_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: metadata.json")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"\nModels saved in: {MODEL_DIR}")
        return True
        
    except Exception as e:
        print(f"ERROR saving models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
