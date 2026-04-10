"""
NVMe Drive Failure Prediction - Flask Dashboard
=================================================
Serves the interactive dashboard and prediction API.
Uses XGBoost model + rule-based failure detection algorithms.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from ml_pipeline import run_all_algorithms, run_independent_algorithms, FAILURE_MODE_LABELS, ALL_FEATURES

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- Load trained artifacts ---
try:
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load models: {e}")
    model = None
    scaler = None
    label_encoder = None

try:
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
except Exception as e:
    print(f"⚠ Warning: Could not load metadata: {e}")
    metadata = {}

app = Flask(__name__, template_folder="templates", static_folder="static")


def build_feature_vector(data: dict) -> np.ndarray:
    """Parse user input and create feature vector using only base features."""
    poh = float(data.get("Power_On_Hours", 0))
    tbw = float(data.get("Total_TBW_TB", 0))
    tbr = float(data.get("Total_TBR_TB", 0))
    temp = float(data.get("Temperature_C", 0))
    life = float(data.get("Percent_Life_Used", 0))
    media = float(data.get("Media_Errors", 0))
    unsafe = float(data.get("Unsafe_Shutdowns", 0))
    crc = float(data.get("CRC_Errors", 0))
    read_err = float(data.get("Read_Error_Rate", 0))
    write_err = float(data.get("Write_Error_Rate", 0))

    # Return only the 10 base features (no engineered features)
    return np.array([[poh, tbw, tbr, temp, life, media, unsafe, crc,
                       read_err, write_err]])


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metadata")
def api_metadata():
    return jsonify(metadata)



@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = build_feature_vector(data)
        
        # Fallback prediction if models didn't load
        if model is None or scaler is None or label_encoder is None:
            metrics = {
                "Power_On_Hours": float(data.get("Power_On_Hours", 0)),
                "Total_TBW_TB": float(data.get("Total_TBW_TB", 0)),
                "Total_TBR_TB": float(data.get("Total_TBR_TB", 0)),
                "Temperature_C": float(data.get("Temperature_C", 0)),
                "Percent_Life_Used": float(data.get("Percent_Life_Used", 0)),
                "Media_Errors": float(data.get("Media_Errors", 0)),
                "Unsafe_Shutdowns": float(data.get("Unsafe_Shutdowns", 0)),
                "CRC_Errors": float(data.get("CRC_Errors", 0)),
                "Read_Error_Rate": float(data.get("Read_Error_Rate", 0)),
                "Write_Error_Rate": float(data.get("Write_Error_Rate", 0)),
            }
            algo_results = run_all_algorithms(metrics)
            top_algo = algo_results[0]
            
            # If top algorithm score is very low, assume No Failure
            if top_algo["score"] < 25:
                pred_mode = 0
                pred_label = "No Failure"
                is_healthy = True
                risk_level = "Low"
            else:
                pred_mode = top_algo["mode"]
                pred_label = FAILURE_MODE_LABELS.get(pred_mode, f"Unknown ({pred_mode})")
                is_healthy = False
                risk_level = "High" if top_algo["score"] >= 75 else "Medium"
            
            # Use rule-based probabilities
            mode_probs = {FAILURE_MODE_LABELS.get(m, f"Mode {m}"): 0.0 for m in FAILURE_MODE_LABELS}
            if top_algo["score"] > 0:
                mode_probs[pred_label] = round(min(0.99, top_algo["score"] / 100.0), 4)
                mode_probs["No Failure"] = round(1.0 - mode_probs[pred_label], 4)
            else:
                mode_probs["No Failure"] = 1.0
        else:
            X_scaled = scaler.transform(X)

            # --- XGBoost prediction ---
            pred_enc = int(model.predict(X_scaled)[0])
            pred_mode = int(label_encoder.inverse_transform([pred_enc])[0])
            pred_label = FAILURE_MODE_LABELS.get(pred_mode, f"Unknown ({pred_mode})")

            proba = model.predict_proba(X_scaled)[0]
            mode_probs = {}
            for i, p in enumerate(proba):
                orig = int(label_encoder.inverse_transform([i])[0])
                mode_probs[FAILURE_MODE_LABELS.get(orig, f"Mode {orig}")] = round(float(p), 4)

            # --- Rule-based algorithm results ---
            metrics = {
                "Power_On_Hours": float(data.get("Power_On_Hours", 0)),
                "Total_TBW_TB": float(data.get("Total_TBW_TB", 0)),
                "Total_TBR_TB": float(data.get("Total_TBR_TB", 0)),
                "Temperature_C": float(data.get("Temperature_C", 0)),
                "Percent_Life_Used": float(data.get("Percent_Life_Used", 0)),
                "Media_Errors": float(data.get("Media_Errors", 0)),
                "Unsafe_Shutdowns": float(data.get("Unsafe_Shutdowns", 0)),
                "CRC_Errors": float(data.get("CRC_Errors", 0)),
                "Read_Error_Rate": float(data.get("Read_Error_Rate", 0)),
                "Write_Error_Rate": float(data.get("Write_Error_Rate", 0)),
            }
            algo_results = run_all_algorithms(metrics)

            # --- Determine overall health status & Risk Level ---
            top_algo = algo_results[0]
            
            # Hybrid Decision: If ML says "No Failure" but rule-based algorithms
            # detect significant degradation (score >= 50), override the ML prediction.
            overridden = False
            if pred_mode == 0 and top_algo["score"] >= 50:
                # Find the failure mode with highest ML probability (excluding No Failure)
                failure_probs = {}
                for i, p in enumerate(proba):
                    orig = int(label_encoder.inverse_transform([i])[0])
                    if orig != 0:
                        failure_probs[orig] = float(p)
                
                # Also consider the rule-based top failure mode
                rule_mode = top_algo["mode"]
                
                # If any failure probability > 5% OR rule score >= 50, override
                best_failure = max(failure_probs, key=failure_probs.get) if failure_probs else rule_mode
                best_failure_prob = failure_probs.get(best_failure, 0)
                
                if best_failure_prob > 0.05 or top_algo["score"] >= 50:
                    pred_mode = rule_mode  # Trust the rules for failure TYPE
                    pred_label = FAILURE_MODE_LABELS.get(pred_mode, f"Unknown ({pred_mode})")
                    overridden = True
                    
            if overridden:
                # Tell the UI the new probability based on the rules override
                forced_prob = min(0.99, top_algo["score"] / 100.0)
                mode_probs["No Failure"] = round(1.0 - forced_prob, 4)
                for k in mode_probs:
                    if k == "No Failure":
                        continue
                    if k == pred_label:
                        mode_probs[k] = round(forced_prob, 4)
                    else:
                        mode_probs[k] = 0.0
                        
            is_healthy = pred_mode == 0 and top_algo["score"] < 25
            
            if not is_healthy:
                risk_level = "High" if (pred_mode != 0 or top_algo["score"] >= 75) else "Medium"
            else:
                risk_level = "Low"
        
        # For ML model path (fallback already set risk_level above)
        if model is not None and scaler is not None and label_encoder is not None:
            risk_level = "Low"
            if not is_healthy:
                risk_level = "High" if (pred_mode != 0 or top_algo["score"] >= 75) else "Medium"
            
        # --- Generate Dynamic Actionable Insights based on Telemetry ---
        factors = []
        actions = []
        
        if metrics["Temperature_C"] >= 75: 
            factors.append(f"Thermal Spike ({metrics['Temperature_C']}°C)")
            actions.append("Check server cooling and chassis airflow. Throttle I/O workload.")
        elif metrics["Temperature_C"] >= 65:
            factors.append("Elevated Temperature")
            
        if metrics["Percent_Life_Used"] >= 90:
            factors.append(f"Critical Wear ({metrics['Percent_Life_Used']}% Life Used)")
            actions.append("Plan drive replacement during next maintenance window. Migrate critical data.")
        elif metrics["Percent_Life_Used"] >= 75:
            factors.append("High Wear-Out")
            
        if metrics["Total_TBW_TB"] > 500:
            factors.append(f"Heavy Write Volume ({int(metrics['Total_TBW_TB'])} TBW)")
            
        errors = metrics["Media_Errors"] + metrics["CRC_Errors"]
        if errors > 0:
            factors.append(f"Data Corruptions ({int(errors)} Errors)")
            actions.append("Identify corrupted logical blocks. Drive replacement highly recommended.")
            
        if metrics["Unsafe_Shutdowns"] >= 10:
            factors.append(f"Frequent Unsafe Shutdowns ({int(metrics['Unsafe_Shutdowns'])})")
            actions.append("Inspect server power delivery and UPS systems.")
            
        if not factors:
            factors = ["No abnormal telemetry signals detected"]
            
        if not actions:
            if is_healthy:
                actions = ["Continue normal operation. Maintain monitoring."]
            else:
                actions = ["Monitor drive closely. Possible unrecognized failure pattern."]
                
        factors_str = " • ".join(factors)
        actions_str = " ".join(actions)

        return jsonify({
            "is_healthy": is_healthy,
            "risk_level": risk_level,
            "insights": {
                "top_contributing_factors": factors_str,
                "suggested_actions": actions_str
            },
            "ml_prediction": {
                "mode": pred_mode,
                "label": pred_label,
                "probabilities": mode_probs,
            },
            "algorithm_results": [
                {
                    "label": a["label"],
                    "mode": a["mode"],
                    "score": a["score"],
                    "reasons": a["reasons"],
                }
                for a in algo_results
            ],
        })

    except Exception as e:
        import traceback
        print(f"ERROR in predict: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/independent-algorithms", methods=["POST"])
def independent_algorithms():
    """
    Returns independent failure detection algorithms (Mode 2 & 3)
    with error percentage filtering (>30% only).
    """
    try:
        data = request.get_json()
        
        metrics = {
            "Power_On_Hours": float(data.get("Power_On_Hours", 0)),
            "Total_TBW_TB": float(data.get("Total_TBW_TB", 0)),
            "Total_TBR_TB": float(data.get("Total_TBR_TB", 0)),
            "Temperature_C": float(data.get("Temperature_C", 0)),
            "Percent_Life_Used": float(data.get("Percent_Life_Used", 0)),
            "Media_Errors": float(data.get("Media_Errors", 0)),
            "Unsafe_Shutdowns": float(data.get("Unsafe_Shutdowns", 0)),
            "CRC_Errors": float(data.get("CRC_Errors", 0)),
            "Read_Error_Rate": float(data.get("Read_Error_Rate", 0)),
            "Write_Error_Rate": float(data.get("Write_Error_Rate", 0)),
        }
        
        # Run independent algorithms
        independent_results = run_independent_algorithms(metrics)
        
        return jsonify({
            "independent_algorithms": [
                {
                    "label": r["label"],
                    "mode": r["mode"],
                    "score": r["score"],
                    "reasons": r["reasons"],
                    "error_filter": r.get("error_filter", "N/A"),
                }
                for r in independent_results
            ],
            "filter_threshold": "Only errors > 30% are shown",
            "total_detected": len(independent_results),
        })
    
    except Exception as e:
        import traceback
        print(f"ERROR in independent_algorithms: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("\n  NVMe Failure Prediction Dashboard")
    print("  http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
