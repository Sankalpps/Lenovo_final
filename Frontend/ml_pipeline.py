"""
NVMe Drive Failure Prediction - ML Pipeline
=============================================
Trains a single XGBoost classifier for multi-class failure mode
classification. Also implements rule-based failure detection algorithms.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "NVMe_Drive_Failure_Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Feature configuration ---
BASE_FEATURES = [
    "Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB",
    "Temperature_C", "Percent_Life_Used", "Media_Errors",
    "Unsafe_Shutdowns", "CRC_Errors", "Read_Error_Rate",
    "Write_Error_Rate"
]
ENGINEERED_FEATURES = ["Error_Rate", "Write_Intensity", "Read_Intensity"]
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

FAILURE_MODE_LABELS = {
    0: "No Failure",
    1: "Wear-Out Failure",
    2: "Thermal Failure",
    3: "Firmware Failure",
    4: "Media Error Failure",
    5: "Unsafe Shutdown Failure",
    6: "Early-Life Failure",
}


# ============================================================
# 1. Rule-Based Failure Detection Algorithms
# ============================================================
# These complement the ML model with domain-expert logic.
# Each algorithm scores the likelihood of a specific failure type.

def detect_wear_out(metrics: dict) -> dict:
    """
    Algorithm 1: Wear-Out Failure Detection
    ----------------------------------------
    Wear-out occurs when a drive has exhausted its write endurance.
    Key indicators: high Percent_Life_Used, high TBW, long power-on hours.
    Scoring formula weights each factor by domain relevance.
    """
    score = 0.0
    reasons = []

    life_used = metrics.get("Percent_Life_Used", 0)
    tbw = metrics.get("Total_TBW_TB", 0)
    poh = metrics.get("Power_On_Hours", 0)

    # Life used > 80% is a strong wear-out indicator
    if life_used >= 90:
        score += 40
        reasons.append(f"Life used critically high at {life_used}% (>90%)")
    elif life_used >= 70:
        score += 25
        reasons.append(f"Life used elevated at {life_used}% (>70%)")
    elif life_used >= 50:
        score += 10
        reasons.append(f"Life used moderate at {life_used}% (>50%)")

    # High TBW indicates heavy write workload
    if tbw >= 300:
        score += 25
        reasons.append(f"Very high total bytes written: {tbw} TB (>300 TB)")
    elif tbw >= 200:
        score += 15
        reasons.append(f"High total bytes written: {tbw} TB (>200 TB)")
    elif tbw >= 150:
        score += 8
        reasons.append(f"Elevated total bytes written: {tbw} TB (>150 TB)")

    # Long power-on hours accelerate wear
    if poh >= 50000:
        score += 20
        reasons.append(f"Extended operation: {poh:,} hours (>50,000 hrs)")
    elif poh >= 35000:
        score += 10
        reasons.append(f"Long operation period: {poh:,} hours (>35,000 hrs)")

    # Write intensity factor
    write_intensity = tbw / max(poh, 1) * 1000
    if write_intensity > 10:
        score += 15
        reasons.append(f"High write intensity: {write_intensity:.2f} GB/hr")

    return {"score": min(score, 100), "reasons": reasons, "mode": 1, "label": "Wear-Out Failure"}


def detect_thermal_failure(metrics: dict) -> dict:
    """
    Algorithm 2: Thermal Failure Detection
    ----------------------------------------
    Thermal failures occur from sustained high temperatures or thermal cycling.
    Key indicators: high Temperature_C, high read/write rates under heat.
    """
    score = 0.0
    reasons = []

    temp = metrics.get("Temperature_C", 0)
    poh = metrics.get("Power_On_Hours", 0)
    read_err = metrics.get("Read_Error_Rate", 0)
    write_err = metrics.get("Write_Error_Rate", 0)

    # Temperature thresholds (NVMe throttles at ~70C, danger at 80C+)
    if temp >= 70:
        score += 45
        reasons.append(f"Critical temperature: {temp}C (>70C - throttling zone)")
    elif temp >= 60:
        score += 30
        reasons.append(f"High temperature: {temp}C (>60C - elevated thermal stress)")
    elif temp >= 55:
        score += 15
        reasons.append(f"Warm temperature: {temp}C (>55C - above optimal)")

    # High error rates at high temperatures compound thermal damage
    if temp >= 55 and (read_err > 15 or write_err > 15):
        score += 20
        reasons.append(f"High I/O errors under thermal stress (Read: {read_err}, Write: {write_err})")
    elif temp >= 50 and (read_err > 20 or write_err > 20):
        score += 10
        reasons.append(f"Elevated I/O errors at warm temperature")

    # Long operation at high temp
    if temp >= 55 and poh >= 30000:
        score += 15
        reasons.append(f"Sustained high-temp operation: {poh:,} hrs at {temp}C")

    return {"score": min(score, 100), "reasons": reasons, "mode": 2, "label": "Thermal Failure"}


def detect_firmware_failure(metrics: dict) -> dict:
    """
    Algorithm 3: Firmware Failure Detection
    ----------------------------------------
    Firmware failures manifest as CRC errors, unexpected shutdowns, and
    anomalous error patterns not explained by hardware wear.
    """
    score = 0.0
    reasons = []

    crc_err = metrics.get("CRC_Errors", 0)
    unsafe = metrics.get("Unsafe_Shutdowns", 0)
    read_err = metrics.get("Read_Error_Rate", 0)
    write_err = metrics.get("Write_Error_Rate", 0)
    media_err = metrics.get("Media_Errors", 0)
    life_used = metrics.get("Percent_Life_Used", 0)

    # CRC errors are a strong firmware/link integrity indicator
    if crc_err >= 5:
        score += 35
        reasons.append(f"High CRC errors: {crc_err} (>5 - data path integrity issue)")
    elif crc_err >= 3:
        score += 20
        reasons.append(f"Elevated CRC errors: {crc_err} (>3)")
    elif crc_err >= 1:
        score += 8
        reasons.append(f"CRC errors detected: {crc_err}")

    # Errors without proportional wear suggest firmware bugs
    if life_used < 30 and (read_err > 15 or write_err > 15):
        score += 25
        reasons.append(f"High error rates ({read_err}/{write_err}) despite low wear ({life_used}%) - firmware anomaly")
    elif life_used < 50 and (read_err > 20 or write_err > 20):
        score += 15
        reasons.append(f"Disproportionate error rates for wear level")

    # Unsafe shutdowns can corrupt firmware state
    if unsafe >= 8:
        score += 20
        reasons.append(f"Many unsafe shutdowns: {unsafe} (>8 - risk of firmware corruption)")
    elif unsafe >= 5:
        score += 10
        reasons.append(f"Multiple unsafe shutdowns: {unsafe}")

    # CRC + no media errors = likely firmware not media
    if crc_err >= 2 and media_err <= 1:
        score += 10
        reasons.append(f"CRC errors without media errors - points to firmware/controller issue")

    return {"score": min(score, 100), "reasons": reasons, "mode": 3, "label": "Firmware Failure"}


def detect_early_life_failure(metrics: dict) -> dict:
    """
    Algorithm 3.5: Early-Life Failure (Rapid Error Accumulation) Detection
    -----------------------------------------------------------------------
    Manufacturing defects cause high error rates in early operation (<3000 hours).
    Indicates likely DOA (Dead On Arrival) or factory defect.
    Key indicators: High errors in early hours, rapid error accumulation pattern.
    """
    score = 0.0
    reasons = []

    poh = metrics.get("Power_On_Hours", 0)
    media_err = metrics.get("Media_Errors", 0)
    crc_err = metrics.get("CRC_Errors", 0)
    read_err = metrics.get("Read_Error_Rate", 0)
    write_err = metrics.get("Write_Error_Rate", 0)
    unsafe = metrics.get("Unsafe_Shutdowns", 0)
    life_used = metrics.get("Percent_Life_Used", 0)

    # Early usage window detection
    if poh < 3000:
        # High error rate in early operation = manufacturing defect signature
        total_errors = media_err + crc_err
        
        # Very early failures (< 100 hours)
        if poh < 100:
            if total_errors >= 1:
                score += 50
                reasons.append(f"Critical: {total_errors} errors detected in first {poh} hours - likely manufacturing defect")
            if read_err > 5 or write_err > 5:
                score += 30
                reasons.append(f"High error rates in initial burn-in period: R={read_err}, W={write_err}")
        
        # Early operation (100-500 hours)
        elif poh < 500:
            if total_errors >= 3:
                score += 40
                reasons.append(f"High error accumulation in early usage: {total_errors} errors at {poh} hours")
            elif total_errors >= 1:
                score += 20
                reasons.append(f"Errors detected early: {total_errors} errors at {poh} hours")
            
            if read_err > 10 or write_err > 10:
                score += 25
                reasons.append(f"Elevated error rates in burn-in: R={read_err}, W={write_err}")
        
        # Early operation (500-3000 hours)
        else:
            if total_errors >= 5:
                score += 35
                reasons.append(f"Rapid error accumulation: {total_errors} errors by {poh} hours")
            elif total_errors >= 2:
                score += 20
                reasons.append(f"Multiple errors in early operation: {total_errors} errors at {poh} hours")
            
            if read_err > 15 or write_err > 15:
                score += 20
                reasons.append(f"Abnormal error rates for this age: R={read_err}, W={write_err}")
        
        # Unsafe shutdowns in early life suggest instability
        if unsafe >= 2:
            score += 15
            reasons.append(f"Instability in early operation: {unsafe} unsafe shutdowns at {poh} hours")
    
    return {"score": min(score, 100), "reasons": reasons, "mode": 6, "label": "Early-Life Failure"}


def detect_media_error_failure(metrics: dict) -> dict:
    """
    Algorithm 4: Media Error Failure Detection
    --------------------------------------------
    Media errors indicate NAND flash cell degradation or bad blocks.
    Key indicators: Media_Errors count, high read/write error rates, SMART flag.
    """
    score = 0.0
    reasons = []

    media_err = metrics.get("Media_Errors", 0)
    read_err = metrics.get("Read_Error_Rate", 0)
    write_err = metrics.get("Write_Error_Rate", 0)
    smart = metrics.get("SMART_Warning_Flag", 0)
    life_used = metrics.get("Percent_Life_Used", 0)

    # Media errors are the primary indicator
    if media_err >= 5:
        score += 40
        reasons.append(f"Critical media errors: {media_err} (>5 - significant cell degradation)")
    elif media_err >= 3:
        score += 25
        reasons.append(f"Elevated media errors: {media_err} (>3)")
    elif media_err >= 2:
        score += 12
        reasons.append(f"Media errors detected: {media_err}")

    # High read error rate suggests read-disturb or cell decay
    if read_err >= 20:
        score += 20
        reasons.append(f"High read error rate: {read_err} (>20 - read head/cell issue)")
    elif read_err >= 12:
        score += 10
        reasons.append(f"Elevated read error rate: {read_err}")

    # Write errors compound media issues
    if write_err >= 15:
        score += 10
        reasons.append(f"High write error rate: {write_err}")

    # SMART warning with media errors is very concerning
    if smart >= 1 and media_err >= 2:
        score += 20
        reasons.append(f"SMART warning active with media errors - drive self-reporting degradation")
    elif smart >= 1:
        score += 10
        reasons.append(f"SMART warning flag active")

    # Media errors at high life usage = expected NAND wear
    if media_err >= 3 and life_used >= 60:
        score += 10
        reasons.append(f"Media errors at {life_used}% life - NAND cell exhaustion")

    return {"score": min(score, 100), "reasons": reasons, "mode": 4, "label": "Media Error Failure"}


def detect_unsafe_shutdown_failure(metrics: dict) -> dict:
    """
    Algorithm 5: Unsafe Shutdown Failure Detection
    ------------------------------------------------
    Repeated unsafe shutdowns damage the FTL mapping table, cause
    data corruption, and can brick drives over time.
    """
    score = 0.0
    reasons = []

    unsafe = metrics.get("Unsafe_Shutdowns", 0)
    poh = metrics.get("Power_On_Hours", 0)
    smart = metrics.get("SMART_Warning_Flag", 0)
    crc_err = metrics.get("CRC_Errors", 0)
    media_err = metrics.get("Media_Errors", 0)

    # Unsafe shutdown count is the primary indicator
    if unsafe >= 10:
        score += 45
        reasons.append(f"Very high unsafe shutdowns: {unsafe} (>10 - severe FTL risk)")
    elif unsafe >= 7:
        score += 30
        reasons.append(f"High unsafe shutdowns: {unsafe} (>7)")
    elif unsafe >= 5:
        score += 18
        reasons.append(f"Elevated unsafe shutdowns: {unsafe} (>5)")
    elif unsafe >= 3:
        score += 8
        reasons.append(f"Multiple unsafe shutdowns: {unsafe}")

    # Shutdown frequency (shutdowns per 1000 hours)
    if poh > 0:
        shutdown_rate = unsafe / poh * 1000
        if shutdown_rate > 0.5:
            score += 20
            reasons.append(f"High shutdown frequency: {shutdown_rate:.2f} per 1000 hrs")
        elif shutdown_rate > 0.2:
            score += 10
            reasons.append(f"Elevated shutdown frequency: {shutdown_rate:.2f} per 1000 hrs")

    # SMART warning with high shutdowns
    if smart >= 1 and unsafe >= 5:
        score += 15
        reasons.append(f"SMART warning with frequent unsafe shutdowns")

    # Unsafe shutdowns causing secondary errors
    if unsafe >= 5 and (crc_err >= 2 or media_err >= 2):
        score += 15
        reasons.append(f"Secondary errors (CRC: {crc_err}, Media: {media_err}) likely caused by power loss events")

    return {"score": min(score, 100), "reasons": reasons, "mode": 5, "label": "Unsafe Shutdown Failure"}


# ============================================================
# Independent Algorithms - Mode 2 & 3 with Error Filtering (>30%)
# ============================================================

def detect_thermal_failure_independent(metrics: dict) -> dict:
    """
    Independent Algorithm: Thermal Failure Detection (Mode 2)
    -----------------------------------------------------------
    Detects thermal failures based on sustained high temperature
    and error correlation. Only reports errors > 30%.
    """
    score = 0.0
    reasons = []
    significant_errors = []

    temp = metrics.get("Temperature_C", 0)
    crc_err = metrics.get("CRC_Errors", 0)
    read_err = metrics.get("Read_Error_Rate", 0)
    write_err = metrics.get("Write_Error_Rate", 0)
    media_err = metrics.get("Media_Errors", 0)

    # Base thermal assessment (0-50 points)
    if temp >= 75:
        score += 50
        reasons.append(f"Critical sustained temperature: {temp}°C (>75°C)")
    elif temp >= 70:
        score += 35
        reasons.append(f"High sustained temperature: {temp}°C (>70°C)")
    elif temp >= 65:
        score += 20
        reasons.append(f"Elevated temperature: {temp}°C (>65°C)")
    elif temp >= 60:
        score += 10
        reasons.append(f"Warm temperature: {temp}°C (>60°C)")

    # Error analysis with 30% threshold filter
    total_errors = crc_err + media_err
    error_threshold = 30
    
    # CRC errors at high temp
    if temp >= 60 and crc_err > 0:
        crc_percentage = (crc_err / max(total_errors, 1)) * 100
        if crc_percentage > error_threshold:
            score += 25
            significant_errors.append(f"CRC Errors: {crc_percentage:.1f}% (>30%)")
    
    # Media errors at high temp
    if temp >= 60 and media_err > 0:
        media_percentage = (media_err / max(total_errors, 1)) * 100
        if media_percentage > error_threshold:
            score += 20
            significant_errors.append(f"Media Errors: {media_percentage:.1f}% (>30%)")
    
    # I/O error rates
    if temp >= 55:
        if read_err > error_threshold:
            score += 15
            significant_errors.append(f"Read Error Rate: {read_err:.1f}% (>30%)")
        if write_err > error_threshold:
            score += 15
            significant_errors.append(f"Write Error Rate: {write_err:.1f}% (>30%)")

    # Combine reasons  
    reasons.extend(significant_errors)
    
    if not significant_errors and score == 0:
        reasons.append("No significant thermal issues detected")

    return {
        "score": min(score, 100),
        "reasons": reasons,
        "mode": 2,
        "label": "Thermal Failure (Independent)",
        "error_filter": "Errors > 30%"
    }


def detect_power_related_failure(metrics: dict) -> dict:
    """
    Independent Algorithm: Power-Related Failure Detection (Mode 3)
    ----------------------------------------------------------------
    Detects power failures based on unsafe shutdowns and corruption errors.
    Only reports errors > 30%.
    """
    score = 0.0
    reasons = []
    significant_errors = []

    unsafe = metrics.get("Unsafe_Shutdowns", 0)
    crc_err = metrics.get("CRC_Errors", 0)
    media_err = metrics.get("Media_Errors", 0)
    poh = metrics.get("Power_On_Hours", 0)

    # Base unsafe shutdown assessment (0-50 points)
    if unsafe >= 10:
        score += 50
        reasons.append(f"Critical unsafe shutdowns: {unsafe} (>10)")
    elif unsafe >= 8:
        score += 35
        reasons.append(f"Very high unsafe shutdowns: {unsafe} (>8)")
    elif unsafe >= 5:
        score += 25
        reasons.append(f"High unsafe shutdowns: {unsafe} (>5)")
    elif unsafe >= 3:
        score += 10
        reasons.append(f"Multiple unsafe shutdowns: {unsafe}")

    # Corruption error analysis with 30% threshold filter
    total_errors = crc_err + media_err
    error_threshold = 30

    # CRC errors (corruption indicator)
    if crc_err > 0:
        crc_percentage = (crc_err / max(total_errors, 1)) * 100
        if crc_percentage > error_threshold:
            score += 25
            significant_errors.append(f"CRC Errors: {crc_percentage:.1f}% (>30%)")
    
    # Media errors (data corruption)
    if media_err > 0:
        media_percentage = (media_err / max(total_errors, 1)) * 100
        if media_percentage > error_threshold:
            score += 20
            significant_errors.append(f"Media Errors: {media_percentage:.1f}% (>30%)")

    # Shutdown frequency
    if poh > 0:
        shutdown_rate = unsafe / poh * 1000
        if shutdown_rate > 0.5:
            score += 10
            reasons.append(f"High shutdown frequency: {shutdown_rate:.2f} per 1000 hrs")

    # Combine reasons
    reasons.extend(significant_errors)
    
    if not significant_errors and score == 0:
        reasons.append("No significant power-related issues detected")

    return {
        "score": min(score, 100),
        "reasons": reasons,
        "mode": 3,
        "label": "Power-Related Failure (Independent)",
        "error_filter": "Errors > 30%"
    }


# ============================================================
# Independent Algorithms - All 5 Failure Modes with Error Filtering (>30%)
# ============================================================

def detect_wearout_failure_independent(metrics: dict) -> dict:
    """
    Independent Algorithm: Wear-Out Failure Detection (Mode 1)
    -----------------------------------------------------------
    Detects wear-out from life exhaustion. Only reports errors > 30%.
    """
    score = 0.0
    reasons = []

    life_used = metrics.get("Percent_Life_Used", 0)
    tbw = metrics.get("Total_TBW_TB", 0)
    poh = metrics.get("Power_On_Hours", 0)

    # Life used assessment
    if life_used >= 95:
        score += 50
        reasons.append(f"Critical life exhaustion: {life_used}% (>95%)")
    elif life_used >= 85:
        score += 35
        reasons.append(f"High life usage: {life_used}% (>85%)")
    elif life_used >= 75:
        score += 20
        reasons.append(f"Elevated life usage: {life_used}% (>75%)")

    # High total bytes written
    if tbw >= 400:
        score += 25
        reasons.append(f"Extreme write volume: {tbw} TB (>400 TB)")
    elif tbw >= 300:
        score += 15
        reasons.append(f"Very high write volume: {tbw} TB (>300 TB)")

    # Extended operation hours
    if poh >= 60000:
        score += 20
        reasons.append(f"Extended operation: {poh:,} hours (>60,000 hrs)")
    elif poh >= 40000:
        score += 10
        reasons.append(f"Long operation: {poh:,} hours (>40,000 hrs)")

    if not reasons:
        reasons.append("No significant wear-out indicators detected")

    return {
        "score": min(score, 100),
        "reasons": reasons,
        "mode": 1,
        "label": "Wear-Out Failure (Independent)",
        "error_filter": "Lifecycle degradation"
    }


def detect_media_error_independent(metrics: dict) -> dict:
    """
    Independent Algorithm: Media Error Failure Detection (Mode 4)
    ---------------------------------------------------------------
    Detects media failures from read/write errors. Only reports errors > 30%.
    """
    score = 0.0
    reasons = []
    significant_errors = []

    media_err = metrics.get("Media_Errors", 0)
    crc_err = metrics.get("CRC_Errors", 0)
    read_err = metrics.get("Read_Error_Rate", 0)
    write_err = metrics.get("Write_Error_Rate", 0)

    # Base media error assessment
    if media_err >= 20:
        score += 50
        reasons.append(f"Critical media errors: {int(media_err)} (>20)")
    elif media_err >= 10:
        score += 30
        reasons.append(f"High media errors: {int(media_err)} (>10)")
    elif media_err >= 5:
        score += 15
        reasons.append(f"Elevated media errors: {int(media_err)} (>5)")

    # Error percentage analysis with 30% threshold
    total_errors = media_err + crc_err
    error_threshold = 30

    # Media errors percentage
    if media_err > 0 and total_errors > 0:
        media_percentage = (media_err / total_errors) * 100
        if media_percentage > error_threshold:
            score += 25
            significant_errors.append(f"Media Errors: {media_percentage:.1f}% (>30%)")

    # Read/Write error rates
    if read_err > error_threshold:
        score += 15
        significant_errors.append(f"Read Error Rate: {read_err:.1f}% (>30%)")
    
    if write_err > error_threshold:
        score += 15
        significant_errors.append(f"Write Error Rate: {write_err:.1f}% (>30%)")

    reasons.extend(significant_errors)

    if not significant_errors and score == 0:
        reasons.append("No significant media errors detected")

    return {
        "score": min(score, 100),
        "reasons": reasons,
        "mode": 4,
        "label": "Media Error Failure (Independent)",
        "error_filter": "Errors > 30%"
    }


def detect_unsafe_shutdown_independent(metrics: dict) -> dict:
    """
    Independent Algorithm: Unsafe Shutdown Failure Detection (Mode 5)
    ------------------------------------------------------------------
    Detects power loss impact from corrupt shutdowns. Only reports errors > 30%.
    """
    score = 0.0
    reasons = []
    significant_errors = []

    unsafe = metrics.get("Unsafe_Shutdowns", 0)
    crc_err = metrics.get("CRC_Errors", 0)
    media_err = metrics.get("Media_Errors", 0)
    poh = metrics.get("Power_On_Hours", 0)

    # Unsafe shutdown assessment
    if unsafe >= 15:
        score += 50
        reasons.append(f"Extreme unsafe shutdowns: {int(unsafe)} (>15)")
    elif unsafe >= 10:
        score += 35
        reasons.append(f"Critical unsafe shutdowns: {int(unsafe)} (>10)")
    elif unsafe >= 5:
        score += 20
        reasons.append(f"High unsafe shutdowns: {int(unsafe)} (>5)")

    # Corruption error analysis with 30% threshold
    total_errors = crc_err + media_err
    error_threshold = 30

    # CRC errors from data corruption
    if crc_err > 0 and total_errors > 0:
        crc_percentage = (crc_err / total_errors) * 100
        if crc_percentage > error_threshold:
            score += 25
            significant_errors.append(f"CRC Errors: {crc_percentage:.1f}% (>30%)")

    # Media errors from incomplete writes
    if media_err > 0 and total_errors > 0:
        media_percentage = (media_err / total_errors) * 100
        if media_percentage > error_threshold:
            score += 20
            significant_errors.append(f"Media Errors: {media_percentage:.1f}% (>30%)")

    # Shutdown frequency
    if poh > 0:
        shutdown_rate = unsafe / poh * 1000
        if shutdown_rate > 1.0:
            score += 15
            reasons.append(f"Extreme shutdown frequency: {shutdown_rate:.2f} per 1000 hrs (>1.0)")
        elif shutdown_rate > 0.5:
            score += 10
            reasons.append(f"High shutdown frequency: {shutdown_rate:.2f} per 1000 hrs (>0.5)")

    reasons.extend(significant_errors)

    if not significant_errors and score == 0:
        reasons.append("No significant unsafe shutdown issues detected")

    return {
        "score": min(score, 100),
        "reasons": reasons,
        "mode": 5,
        "label": "Unsafe Shutdown Failure (Independent)",
        "error_filter": "Errors > 30%"
    }


def run_all_algorithms(metrics: dict) -> list:
    """Run all failure detection algorithms and return sorted results."""
    results = [
        detect_wear_out(metrics),
        detect_thermal_failure(metrics),
        detect_firmware_failure(metrics),
        detect_early_life_failure(metrics),
        detect_media_error_failure(metrics),
        detect_unsafe_shutdown_failure(metrics),
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def run_independent_algorithms(metrics: dict) -> list:
    """Run all independent failure detection algorithms (Mode 1, 2, 3, 4, 5) with error filtering."""
    results = [
        detect_wearout_failure_independent(metrics),
        detect_thermal_failure_independent(metrics),
        detect_power_related_failure(metrics),
        detect_media_error_independent(metrics),
        detect_unsafe_shutdown_independent(metrics),
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    # Filter to only include results with scores > 0
    return [r for r in results if r["score"] > 0]


# ============================================================
# 2. Data Loading & Feature Engineering
# ============================================================

def load_and_engineer(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Handle missing values
    for col in BASE_FEATURES:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Engineered features (safe division)
    poh = df["Power_On_Hours"].replace(0, np.nan)
    df["Error_Rate"] = (df["Media_Errors"] + df["CRC_Errors"]) / poh
    df["Write_Intensity"] = df["Total_TBW_TB"] / poh
    df["Read_Intensity"] = df["Total_TBR_TB"] / poh
    for feat in ENGINEERED_FEATURES:
        df[feat].fillna(0, inplace=True)
        df[feat].replace([np.inf, -np.inf], 0, inplace=True)

    return df


# ============================================================
# 3. Train XGBoost Model
# ============================================================

def train_model():
    print("=" * 60)
    print("  NVMe Failure Mode Classification - XGBoost Training")
    print("=" * 60)

    df = load_and_engineer()
    print(f"\n  Dataset: {df.shape[0]} samples loaded for training")
    print(f"  Features: {len(ALL_FEATURES)}")

    X = df[ALL_FEATURES].values
    y = df["Failure_Mode"].values

    # Encode labels to be contiguous for XGBoost
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"  Classes: {dict(zip(le.classes_, [FAILURE_MODE_LABELS.get(c, c) for c in le.classes_]))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # XGBoost - best model for tabular imbalanced classification
    # Class balancing: failure cases are rarer but critical to detect
    sample_weights = compute_sample_weight('balanced', y_train)
    print("\n  Training XGBoost classifier (class-balanced)...")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train, sample_weight=sample_weights)

    # Evaluate
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Test Accuracy:  {acc:.4f}")
    print(f"  Test F1-Score:  {f1:.4f}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X_train_s, y_train, cv=cv,
                                scoring=["accuracy", "f1_weighted"])
    cv_acc = cv_results["test_accuracy"].mean()
    cv_f1 = cv_results["test_f1_weighted"].mean()
    print(f"  CV Accuracy:    {cv_acc:.4f}")
    print(f"  CV F1-Score:    {cv_f1:.4f}")

    # Feature importance
    fi = sorted(
        zip(ALL_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n  Feature Importance Ranking:")
    for i, (feat, imp) in enumerate(fi, 1):
        bar = "#" * int(imp * 100)
        print(f"    {i:2d}. {feat:22s} {imp:.4f}  {bar}")

    # Save artifacts
    print("\n  Saving model artifacts...")
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    meta = {
        "model": "XGBoost",
        "features": ALL_FEATURES,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "cv_accuracy": round(cv_acc, 4),
        "cv_f1_score": round(cv_f1, 4),
        "feature_importance": [
            {"feature": f, "importance": round(float(v), 6)} for f, v in fi
        ],
        "failure_mode_labels": {str(k): v for k, v in FAILURE_MODE_LABELS.items()},
        "label_classes": le.classes_.tolist(),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("  Done! Run `python app.py` to launch the dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
