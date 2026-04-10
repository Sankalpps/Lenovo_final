"""
Enhanced Training Script: Early-Life Failure Detection
======================================================
Identifies manufacturing defects (early-life failures) in drives with
high error rates in early operation (<3000 hours)
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
    6: "Early-Life Failure",
}

def identify_early_life_failures(df):
    """
    Identify manufacturing defects based on error patterns in early operation.
    Returns updated dataframe with Early-Life Failure labels.
    """
    print("\n[Enhanced] Identifying Early-Life Failures (Manufacturing Defects)...")
    
    early_life_count = 0
    
    # Check drives with < 3000 hours for manufacturing defects
    early_drives = df[df['Power_On_Hours'] < 3000].copy()
    total_early = len(early_drives)
    
    print(f"  Found {total_early} drives with <3000 POH (early operation)")
    
    # Identify manufacturing defects: high errors in very early operation
    for idx in early_drives.index:
        poh = df.loc[idx, 'Power_On_Hours']
        media_err = df.loc[idx, 'Media_Errors']
        crc_err = df.loc[idx, 'CRC_Errors']
        read_err = df.loc[idx, 'Read_Error_Rate']
        write_err = df.loc[idx, 'Write_Error_Rate']
        unsafe = df.loc[idx, 'Unsafe_Shutdowns']
        
        total_errors = media_err + crc_err
        is_early_defect = False
        
        # Very early failures (< 100 hours)
        if poh < 100 and total_errors >= 1:
            is_early_defect = True
        # Early burns-in (< 500 hours) 
        elif poh < 500 and total_errors >= 3:
            is_early_defect = True
        # General early operation
        elif poh < 3000 and total_errors >= 5 and (read_err > 15 or write_err > 15):
            is_early_defect = True
        # Instability in early life
        elif poh < 500 and unsafe >= 2 and (read_err > 10 or write_err > 10):
            is_early_defect = True
        
        if is_early_defect:
            df.loc[idx, 'Failure_Mode'] = 6
            early_life_count += 1
    
    print(f"  Identified {early_life_count} manufacturing defects (Early-Life Failures)")
    return df

def train_enhanced_model():
    print("="*80)
    print("ENHANCED TRAINING: NVME FAILURE PREDICTION WITH EARLY-LIFE DETECTION")
    print("="*80)
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return False
    
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} samples")
    
    # Identify early-life failures
    print("\n[2/6] Enhanced Labeling...")
    df = identify_early_life_failures(df)
    
    print(f"\n  Updated Failure Mode Distribution:")
    for mode in sorted(df['Failure_Mode'].unique()):
        count = len(df[df['Failure_Mode'] == mode])
        label = FAILURE_MODE_LABELS.get(mode, f"Unknown {mode}")
        print(f"    - {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Prepare data
    print("\n[3/6] Preparing features...")
    X = df[BASE_FEATURES].values
    y = df["Failure_Mode"].values
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Split data
    print("\n[4/6] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Preprocess
    print("\n[5/6] Preprocessing...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"  Label Mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Train model
    print("\n[6/6] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=250,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0,
        tree_method='hist',
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(
        X_train_scaled, y_train_encoded,
        eval_set=[(X_test_scaled, y_test_encoded)],
        verbose=False
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("ENHANCED MODEL EVALUATION")
    print("="*80)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train_encoded, y_pred_train)
    test_acc = accuracy_score(y_test_encoded, y_pred_test)
    
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc*100:.2f}%")
    print(f"  Test:  {test_acc*100:.2f}%")
    
    print(f"\nWeighted Metrics:")
    print(f"  Precision: {precision_score(y_test_encoded, y_pred_test, average='weighted')*100:.2f}%")
    print(f"  Recall:    {recall_score(y_test_encoded, y_pred_test, average='weighted')*100:.2f}%")
    print(f"  F1-Score:  {f1_score(y_test_encoded, y_pred_test, average='weighted')*100:.2f}%")
    
    print(f"\nClassification Report:")
    report = classification_report(
        y_test_encoded, y_pred_test,
        target_names=[FAILURE_MODE_LABELS.get(mode, f"Mode {mode}") for mode in label_encoder.classes_],
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Save models
    print("\n" + "="*80)
    print("SAVING ENHANCED MODELS")
    print("="*80)
    
    try:
        joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
        print(f"  Saved: xgb_model.pkl (Enhanced with Early-Life Detection)")
        
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
            "test_samples": len(X_test),
            "enhancement": "Added Early-Life Failure (Manufacturing Defect) Detection"
        }
        
        with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: metadata.json")
        
        print("\n" + "="*80)
        print("ENHANCED TRAINING COMPLETE!")
        print("="*80)
        print(f"\nNew Failure Mode Added: Mode 6 - Early-Life Failure (Manufacturing Defects)")
        print(f"Models saved in: {MODEL_DIR}")
        return True
        
    except Exception as e:
        print(f"ERROR saving models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_enhanced_model()
    sys.exit(0 if success else 1)
