"""
NVMe Binary Failure Classifier — Modes 1 & 4
==============================================
Predicts whether a drive will experience:
  - Failure Mode 1 (Wear-Out Failure)
  - Failure Mode 4 (Media Error Failure)

Uses XGBoost with SMOTE oversampling to handle extreme class imbalance.

Usage:
    python binary_classifier.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imblearn not installed. Install with: pip install imbalanced-learn")
    print("       Falling back to class_weight balancing only.\n")

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "NVMe_Drive_Failure_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "binary_model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features to use (all relevant SMART metrics, NO identifiers)
DROP_COLS = ["Drive_ID", "Vendor", "Model", "Firmware_Version",
             "Failure_Mode", "Failure_Flag", "SMART_Warning_Flag"]

TARGET_FAILURE_MODES = [1, 4]  # Wear-Out and Media Error


# ============================================================
# 1. Data Loading & Preprocessing
# ============================================================
def load_and_preprocess():
    """Load dataset, clean it, engineer features, create binary target."""
    print("=" * 60)
    print("  STEP 1: Data Loading & Preprocessing")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"  Raw dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- Remove duplicates ---
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        print(f"  Removed {dupes} duplicate rows")
    else:
        print(f"  No duplicates found")

    # --- Handle missing values (median imputation for numerics) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_count = df[numeric_cols].isnull().sum().sum()
    if missing_count > 0:
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} missing values with median={median_val:.2f}")
    else:
        print(f"  No missing values found")

    # --- Original failure mode distribution ---
    print(f"\n  Original Failure_Mode distribution:")
    for mode, count in df["Failure_Mode"].value_counts().sort_index().items():
        label = {0: "No Failure", 1: "Wear-Out", 4: "Media Error", 5: "Unsafe Shutdown"}.get(mode, f"Mode {mode}")
        marker = " <-- TARGET" if mode in TARGET_FAILURE_MODES else ""
        print(f"    Mode {mode} ({label}): {count:,}{marker}")

    # --- Create binary target ---
    # Modes 1 & 4 -> class 1 (failure), everything else -> class 0 (healthy)
    df["target"] = df["Failure_Mode"].apply(lambda x: 1 if x in TARGET_FAILURE_MODES else 0)
    pos = df["target"].sum()
    neg = len(df) - pos
    print(f"\n  Binary target created:")
    print(f"    Class 0 (Healthy):  {neg:,} ({neg/len(df)*100:.1f}%)")
    print(f"    Class 1 (Failure):  {pos:,} ({pos/len(df)*100:.1f}%)")
    print(f"    Imbalance ratio:    1:{neg//pos}")

    # --- Feature engineering ---
    poh = df["Power_On_Hours"].replace(0, np.nan)
    df["Error_Rate"] = (df["Media_Errors"] + df["CRC_Errors"]) / poh
    df["Write_Intensity"] = df["Total_TBW_TB"] / poh
    df["Read_Intensity"] = df["Total_TBR_TB"] / poh
    df["TBW_per_Life"] = df["Total_TBW_TB"] / df["Percent_Life_Used"].replace(0, np.nan)
    df["Temp_x_Hours"] = df["Temperature_C"] * df["Power_On_Hours"] / 1000  # thermal exposure
    df["Total_Errors"] = df["Media_Errors"] + df["CRC_Errors"] + df["Unsafe_Shutdowns"]

    # Clean infinities and NaNs from engineered features
    eng_features = ["Error_Rate", "Write_Intensity", "Read_Intensity",
                    "TBW_per_Life", "Temp_x_Hours", "Total_Errors"]
    for feat in eng_features:
        df[feat].fillna(0, inplace=True)
        df[feat].replace([np.inf, -np.inf], 0, inplace=True)

    # --- Drop irrelevant columns ---
    feature_cols = [c for c in df.columns if c not in DROP_COLS + ["target"] + eng_features
                    if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    feature_cols += eng_features
    print(f"\n  Final feature set ({len(feature_cols)} features):")
    for f in feature_cols:
        print(f"    - {f}")

    X = df[feature_cols].values
    y = df["target"].values

    return X, y, feature_cols, df


# ============================================================
# 2. Model Training with SMOTE + XGBoost
# ============================================================
def train_binary_model(X, y, feature_names):
    """Train XGBoost binary classifier with SMOTE oversampling."""
    print("\n" + "=" * 60)
    print("  STEP 2: Model Training")
    print("=" * 60)

    # --- Train/Test split (80/20, stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train set: {X_train.shape[0]:,} samples ({y_train.sum()} failures)")
    print(f"  Test set:  {X_test.shape[0]:,} samples ({y_test.sum()} failures)")

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Apply SMOTE to training data ---
    if HAS_SMOTE:
        print(f"\n  Applying SMOTE oversampling...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_s, y_train)
        print(f"  Before SMOTE: {len(y_train):,} samples ({y_train.sum()} failures)")
        print(f"  After SMOTE:  {len(y_train_resampled):,} samples ({y_train_resampled.sum()} failures)")
    else:
        X_train_resampled, y_train_resampled = X_train_s, y_train
        print("  Skipping SMOTE (not installed)")

    # --- Calculate scale_pos_weight for additional class balancing ---
    neg_count = (y_train_resampled == 0).sum()
    pos_count = (y_train_resampled == 1).sum()
    scale_weight = neg_count / max(pos_count, 1)

    # --- XGBoost Binary Classifier ---
    print(f"\n  Training XGBoost Binary Classifier...")
    print(f"  scale_pos_weight: {scale_weight:.2f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_resampled, y_train_resampled)

    return model, scaler, X_train_s, X_test_s, y_train, y_test


# ============================================================
# 3. Evaluation
# ============================================================
def evaluate_model(model, X_test_s, y_test, feature_names):
    """Full evaluation: accuracy, precision, recall, F1, confusion matrix, ROC."""
    print("\n" + "=" * 60)
    print("  STEP 3: Model Evaluation")
    print("=" * 60)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    # --- Core Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0

    print(f"\n  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}  <-- CRITICAL for failure detection")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  ROC AUC:    {auc:.4f}")

    # --- Classification Report ---
    print(f"\n  Classification Report:")
    target_names = ["Healthy (0)", "Failure 1|4 (1)"]
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print(report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                  Predicted Healthy  Predicted Failure")
    print(f"  Actual Healthy       {cm[0][0]:>6}           {cm[0][1]:>6}")
    print(f"  Actual Failure       {cm[1][0]:>6}           {cm[1][1]:>6}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives (caught failures):  {tp}")
    print(f"  False Negatives (missed failures): {fn}  <-- want this to be 0")
    print(f"  False Positives (false alarms):    {fp}")
    print(f"  True Negatives (correct healthy):  {tn}")

    # --- Save Confusion Matrix Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix — Failure Modes 1 & 4", fontsize=15)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {cm_path}")

    # --- ROC Curve ---
    if auc > 0:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(fpr, tpr, color="#00d4ff", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
        ax2.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
        ax2.fill_between(fpr, tpr, alpha=0.15, color="#00d4ff")
        ax2.set_xlabel("False Positive Rate", fontsize=13)
        ax2.set_ylabel("True Positive Rate (Recall)", fontsize=13)
        ax2.set_title("ROC Curve — Failure Modes 1 & 4 Detection", fontsize=15)
        ax2.legend(loc="lower right", fontsize=12)
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
        fig2.savefig(roc_path, dpi=150)
        plt.close(fig2)
        print(f"  Saved: {roc_path}")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# 4. Feature Importance
# ============================================================
def plot_feature_importance(model, feature_names):
    """Plot and save feature importance chart."""
    print("\n" + "=" * 60)
    print("  STEP 4: Feature Importance Analysis")
    print("=" * 60)

    importances = model.feature_importances_
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  Top Features Contributing to Failures 1 & 4:")
    print(f"  {'Rank':<5} {'Feature':<25} {'Importance':<12} {'Bar'}")
    print(f"  {'-'*5} {'-'*25} {'-'*12} {'-'*30}")
    for i, (feat, imp) in enumerate(fi, 1):
        bar = "█" * int(imp * 80)
        print(f"  {i:<5} {feat:<25} {imp:<12.6f} {bar}")

    # --- Feature Importance Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    features, values = zip(*fi)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=13)
    ax.set_title("Feature Importance — Predicting Failure Modes 1 & 4", fontsize=15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {fi_path}")

    return fi


# ============================================================
# 5. Failure Pattern Insights
# ============================================================
def analyze_failure_patterns(df):
    """Analyze patterns in failure modes 1 and 4."""
    print("\n" + "=" * 60)
    print("  STEP 5: Failure Pattern Insights")
    print("=" * 60)

    healthy = df[df["target"] == 0]
    failed = df[df["target"] == 1]
    mode1 = df[df["Failure_Mode"] == 1]
    mode4 = df[df["Failure_Mode"] == 4]

    compare_cols = ["Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB",
                    "Temperature_C", "Percent_Life_Used", "Media_Errors",
                    "Unsafe_Shutdowns", "CRC_Errors", "Read_Error_Rate",
                    "Write_Error_Rate"]

    print(f"\n  === Mode 1 (Wear-Out Failure) — {len(mode1)} samples ===")
    print(f"  {'Metric':<22} {'Healthy Mean':>14} {'Mode 1 Mean':>14} {'Difference':>12}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*12}")
    for col in compare_cols:
        h_mean = healthy[col].mean()
        f_mean = mode1[col].mean()
        diff = f_mean - h_mean
        arrow = "▲" if diff > 0 else "▼" if diff < 0 else "="
        print(f"  {col:<22} {h_mean:>14.2f} {f_mean:>14.2f} {arrow} {abs(diff):>10.2f}")

    print(f"\n  === Mode 4 (Media Error Failure) — {len(mode4)} samples ===")
    print(f"  {'Metric':<22} {'Healthy Mean':>14} {'Mode 4 Mean':>14} {'Difference':>12}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*12}")
    for col in compare_cols:
        h_mean = healthy[col].mean()
        f_mean = mode4[col].mean()
        diff = f_mean - h_mean
        arrow = "▲" if diff > 0 else "▼" if diff < 0 else "="
        print(f"  {col:<22} {h_mean:>14.2f} {f_mean:>14.2f} {arrow} {abs(diff):>10.2f}")

    # --- Key Insights ---
    print(f"\n  === KEY INSIGHTS ===")

    # Mode 1 insights
    if len(mode1) > 0:
        print(f"\n  Wear-Out Failure (Mode 1) Patterns:")
        avg_life = mode1["Percent_Life_Used"].mean()
        avg_tbw = mode1["Total_TBW_TB"].mean()
        avg_poh = mode1["Power_On_Hours"].mean()
        print(f"    - Average life used: {avg_life:.1f}% (vs {healthy['Percent_Life_Used'].mean():.1f}% healthy)")
        print(f"    - Average TBW: {avg_tbw:.1f} TB (vs {healthy['Total_TBW_TB'].mean():.1f} TB healthy)")
        print(f"    - Average Power-On Hours: {avg_poh:,.0f} hrs")
        print(f"    => Heavy write workloads on aged drives are the primary trigger")

    # Mode 4 insights
    if len(mode4) > 0:
        print(f"\n  Media Error Failure (Mode 4) Patterns:")
        avg_media = mode4["Media_Errors"].mean()
        avg_read = mode4["Read_Error_Rate"].mean()
        avg_write = mode4["Write_Error_Rate"].mean()
        print(f"    - Average media errors: {avg_media:.1f} (vs {healthy['Media_Errors'].mean():.1f} healthy)")
        print(f"    - Average read error rate: {avg_read:.1f} (vs {healthy['Read_Error_Rate'].mean():.1f} healthy)")
        print(f"    - Average write error rate: {avg_write:.1f} (vs {healthy['Write_Error_Rate'].mean():.1f} healthy)")
        print(f"    => NAND flash cell degradation with high I/O error rates")


# ============================================================
# 6. Prediction Function for New Data
# ============================================================
def predict_new_drive(model, scaler, feature_names, drive_data: dict):
    """
    Predict failure for a new drive.

    Parameters:
        drive_data: dict with keys matching feature_names
        Example: {"Power_On_Hours": 35000, "Total_TBW_TB": 400, ...}

    Returns:
        dict with prediction, probability, and risk assessment
    """
    # Build feature vector
    raw_values = []
    for feat in feature_names:
        if feat in drive_data:
            raw_values.append(float(drive_data[feat]))
        elif feat == "Error_Rate":
            poh = max(float(drive_data.get("Power_On_Hours", 1)), 1)
            raw_values.append((float(drive_data.get("Media_Errors", 0)) +
                             float(drive_data.get("CRC_Errors", 0))) / poh)
        elif feat == "Write_Intensity":
            poh = max(float(drive_data.get("Power_On_Hours", 1)), 1)
            raw_values.append(float(drive_data.get("Total_TBW_TB", 0)) / poh)
        elif feat == "Read_Intensity":
            poh = max(float(drive_data.get("Power_On_Hours", 1)), 1)
            raw_values.append(float(drive_data.get("Total_TBR_TB", 0)) / poh)
        elif feat == "TBW_per_Life":
            life = max(float(drive_data.get("Percent_Life_Used", 1)), 1)
            raw_values.append(float(drive_data.get("Total_TBW_TB", 0)) / life)
        elif feat == "Temp_x_Hours":
            raw_values.append(float(drive_data.get("Temperature_C", 0)) *
                            float(drive_data.get("Power_On_Hours", 0)) / 1000)
        elif feat == "Total_Errors":
            raw_values.append(float(drive_data.get("Media_Errors", 0)) +
                            float(drive_data.get("CRC_Errors", 0)) +
                            float(drive_data.get("Unsafe_Shutdowns", 0)))
        else:
            raw_values.append(0.0)

    X = np.array([raw_values])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    failure_prob = proba[1]

    result = {
        "prediction": "FAILURE RISK (Mode 1 or 4)" if pred == 1 else "HEALTHY",
        "failure_probability": f"{failure_prob*100:.1f}%",
        "confidence": f"{max(proba)*100:.1f}%",
        "risk_level": (
            "CRITICAL" if failure_prob >= 0.75 else
            "HIGH" if failure_prob >= 0.50 else
            "MEDIUM" if failure_prob >= 0.25 else
            "LOW"
        ),
    }
    return result


# ============================================================
# 7. Cross-Validation
# ============================================================
def cross_validate_model(model, X, y):
    """5-fold stratified cross-validation."""
    print("\n" + "=" * 60)
    print("  STEP 6: Cross-Validation (5-Fold Stratified)")
    print("=" * 60)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X_s, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False
    )

    print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        key = f"test_{metric}"
        mean = scores[key].mean()
        std = scores[key].std()
        print(f"  {metric:<15} {mean:>8.4f} {std:>8.4f}")

    return scores


# ============================================================
# Main Pipeline
# ============================================================
def run_pipeline():
    """Execute the complete binary classification pipeline."""
    print("\n")
    print("*" * 60)
    print("  NVMe BINARY FAILURE CLASSIFIER")
    print("  Target: Failure Modes 1 (Wear-Out) & 4 (Media Error)")
    print("*" * 60)

    # 1. Load & preprocess
    X, y, feature_names, df = load_and_preprocess()

    # 2. Train model
    model, scaler, X_train_s, X_test_s, y_train, y_test = train_binary_model(X, y, feature_names)

    # 3. Evaluate
    metrics = evaluate_model(model, X_test_s, y_test, feature_names)

    # 4. Feature importance
    fi = plot_feature_importance(model, feature_names)

    # 5. Pattern insights
    analyze_failure_patterns(df)

    # 6. Cross-validation
    cv_scores = cross_validate_model(model, X, y)

    # 7. Save model artifacts
    print("\n" + "=" * 60)
    print("  STEP 7: Saving Artifacts")
    print("=" * 60)

    joblib.dump(model, os.path.join(OUTPUT_DIR, "binary_xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "binary_scaler.pkl"))

    meta = {
        "model": "XGBoost Binary Classifier",
        "target": "Failure Modes 1 (Wear-Out) & 4 (Media Error)",
        "features": feature_names,
        "metrics": metrics,
        "cv_recall_mean": round(float(cv_scores["test_recall"].mean()), 4),
        "cv_f1_mean": round(float(cv_scores["test_f1"].mean()), 4),
        "feature_importance": [
            {"feature": f, "importance": round(float(v), 6)} for f, v in fi
        ],
    }
    with open(os.path.join(OUTPUT_DIR, "binary_model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved:     {OUTPUT_DIR}/binary_xgb_model.pkl")
    print(f"  Scaler saved:    {OUTPUT_DIR}/binary_scaler.pkl")
    print(f"  Metadata saved:  {OUTPUT_DIR}/binary_model_metadata.json")
    print(f"  Plots saved:     {OUTPUT_DIR}/")

    # 8. Demo prediction
    print("\n" + "=" * 60)
    print("  DEMO: Predicting New Drive Data")
    print("=" * 60)

    # Test Case 1: Healthy drive
    healthy_drive = {
        "Power_On_Hours": 5000, "Total_TBW_TB": 30, "Total_TBR_TB": 50,
        "Temperature_C": 35, "Percent_Life_Used": 5, "Media_Errors": 0,
        "Unsafe_Shutdowns": 0, "CRC_Errors": 0, "Read_Error_Rate": 1,
        "Write_Error_Rate": 0.5,
    }
    result1 = predict_new_drive(model, scaler, feature_names, healthy_drive)
    print(f"\n  Test 1 — Healthy Drive (5K hrs, 30 TBW, low errors):")
    for k, v in result1.items():
        print(f"    {k}: {v}")

    # Test Case 2: Wear-out risk (Mode 1)
    wearout_drive = {
        "Power_On_Hours": 55000, "Total_TBW_TB": 450, "Total_TBR_TB": 300,
        "Temperature_C": 45, "Percent_Life_Used": 92, "Media_Errors": 3,
        "Unsafe_Shutdowns": 2, "CRC_Errors": 1, "Read_Error_Rate": 12,
        "Write_Error_Rate": 8,
    }
    result2 = predict_new_drive(model, scaler, feature_names, wearout_drive)
    print(f"\n  Test 2 — Worn-Out Drive (55K hrs, 450 TBW, 92% life):")
    for k, v in result2.items():
        print(f"    {k}: {v}")

    # Test Case 3: Media error risk (Mode 4)
    media_drive = {
        "Power_On_Hours": 10000, "Total_TBW_TB": 80, "Total_TBR_TB": 100,
        "Temperature_C": 40, "Percent_Life_Used": 15, "Media_Errors": 8,
        "Unsafe_Shutdowns": 1, "CRC_Errors": 4, "Read_Error_Rate": 25,
        "Write_Error_Rate": 18,
    }
    result3 = predict_new_drive(model, scaler, feature_names, media_drive)
    print(f"\n  Test 3 — Media Error Drive (8 media errors, high I/O errors):")
    for k, v in result3.items():
        print(f"    {k}: {v}")

    print("\n" + "*" * 60)
    print("  PIPELINE COMPLETE")
    print("*" * 60)


if __name__ == "__main__":
    run_pipeline()
