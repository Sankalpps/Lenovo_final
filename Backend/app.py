from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
import sqlite3

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = '../ml_training/trained_models'
DATA_DIR = '../ml_training/processed_data'
DB_PATH = './predictions.db'

# Global variables for models and preprocessing
models = {}
encoders = {}
scaler = None
feature_names = None


def load_models():
    """Load all trained models"""
    global models
    print("Loading models...")
    
    model_files = [
        'logistic_regression_model.pkl',
        'random_forest_model.pkl',
        'gradient_boosting_model.pkl',
        'xgboost_model.pkl',
        'svm_model.pkl'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_name = model_file.replace('_model.pkl', '')
                models[model_name] = pickle.load(f)
                print(f"  Loaded: {model_name}")
        else:
            print(f"  WARNING: {model_file} not found")
    
    if not models:
        print("  WARNING: No models loaded!")


def load_preprocessing_tools():
    """Load preprocessing tools (encoders, scaler)"""
    global encoders, scaler, feature_names
    print("Loading preprocessing tools...")
    
    # Load encoders
    encoders_path = os.path.join(DATA_DIR, 'encoders.pkl')
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            print(f"  Loaded encoders: {list(encoders.keys())}")
    
    # Load scaler
    scaler_path = os.path.join(DATA_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            print(f"  Loaded scaler")
    
    # Feature names from training
    feature_names = [
        'Vendor', 'Model', 'Firmware_Version', 'Power_On_Hours',
        'Total_TBW_TB', 'Total_TBR_TB', 'Temperature_C',
        'Percent_Life_Used', 'Media_Errors', 'Unsafe_Shutdowns',
        'CRC_Errors', 'Read_Error_Rate', 'Write_Error_Rate',
        'SMART_Warning_Flag', 'Power_Temp_Ratio', 'Error_Sum',
        'Error_Rate_Sum', 'Wear_Temp_Ratio'
    ]
    
    print(f"  Loaded feature names: {len(feature_names)} features")


def init_db():
    """Initialize SQLite database"""
    if os.path.exists(DB_PATH):
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            drive_id TEXT,
            model_name TEXT,
            prediction INTEGER,
            probability REAL,
            input_data TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            total_predictions INTEGER,
            failure_predictions INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized")


def save_prediction(drive_id, model_name, prediction, probability, input_data):
    """Save prediction to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    input_json = json.dumps(input_data)
    
    cursor.execute('''
        INSERT INTO predictions (timestamp, drive_id, model_name, prediction, probability, input_data)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, drive_id, model_name, prediction, probability, input_json))
    
    conn.commit()
    conn.close()


def preprocess_input(data):
    """Preprocess input data for prediction"""
    global encoders, scaler
    
    # Create a copy to avoid modifying original
    processed = data.copy()
    
    # Encode categorical features
    if 'Vendor' in processed:
        if 'Vendor' in encoders:
            try:
                processed['Vendor'] = encoders['Vendor'].transform([processed['Vendor']])[0]
            except:
                processed['Vendor'] = 0
    
    if 'Model' in processed:
        if 'Model' in encoders:
            try:
                processed['Model'] = encoders['Model'].transform([processed['Model']])[0]
            except:
                processed['Model'] = 0
    
    if 'Firmware_Version' in processed:
        if 'Firmware_Version' in encoders:
            try:
                processed['Firmware_Version'] = encoders['Firmware_Version'].transform([processed['Firmware_Version']])[0]
            except:
                processed['Firmware_Version'] = 0
    
    # Create engineered features
    processed['Power_Temp_Ratio'] = processed['Power_On_Hours'] / (processed['Temperature_C'] + 1)
    processed['Error_Sum'] = (processed['Media_Errors'] + 
                             processed['Unsafe_Shutdowns'] + 
                             processed['CRC_Errors'])
    processed['Error_Rate_Sum'] = processed['Read_Error_Rate'] + processed['Write_Error_Rate']
    processed['Wear_Temp_Ratio'] = processed['Percent_Life_Used'] / (processed['Temperature_C'] + 1)
    
    # Extract features in correct order
    X = []
    for feature in feature_names:
        X.append(processed.get(feature, 0))
    
    X = np.array([X])
    
    # Scale
    if scaler:
        X = scaler.transform(X)
    
    return X


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """List available models"""
    return jsonify({
        'models': list(models.keys()),
        'count': len(models)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on input data"""
    
    try:
        data = request.json
        
        # Validate input
        required_fields = [
            'Vendor', 'Model', 'Firmware_Version', 'Power_On_Hours',
            'Total_TBW_TB', 'Total_TBR_TB', 'Temperature_C',
            'Percent_Life_Used', 'Media_Errors', 'Unsafe_Shutdowns',
            'CRC_Errors', 'Read_Error_Rate', 'Write_Error_Rate',
            'SMART_Warning_Flag'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing fields: {missing}',
                'required_fields': required_fields
            }), 400
        
        # Preprocess input
        X = preprocess_input(data)
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in models.items():
            pred = model.predict(X)[0]
            pred_proba = model.predict_proba(X)[0]
            probability = float(pred_proba[1])  # Probability of failure
            
            predictions[model_name] = {
                'prediction': int(pred),
                'probability': probability,
                'confidence': float(max(pred_proba))
            }
            
            # Save to database
            save_prediction(
                data.get('Drive_ID', 'unknown'),
                model_name,
                int(pred),
                probability,
                data
            )
        
        # Calculate ensemble prediction (majority vote)
        ensemble_pred = np.mean([p['prediction'] for p in predictions.values()])
        ensemble_prob = np.mean([p['probability'] for p in predictions.values()])
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'ensemble': {
                'prediction': int(round(ensemble_pred)),
                'probability': float(ensemble_prob)
            },
            'failure_risk': 'HIGH' if ensemble_prob > 0.6 else ('MEDIUM' if ensemble_prob > 0.3 else 'LOW')
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Make predictions on multiple records"""
    
    try:
        data = request.json
        records = data.get('records', [])
        
        if not records:
            return jsonify({'error': 'No records provided'}), 400
        
        results = []
        for record in records:
            try:
                X = preprocess_input(record)
                
                predictions = {}
                for model_name, model in models.items():
                    pred = model.predict(X)[0]
                    pred_proba = model.predict_proba(X)[0]
                    probability = float(pred_proba[1])
                    
                    predictions[model_name] = {
                        'prediction': int(pred),
                        'probability': probability
                    }
                
                ensemble_pred = np.mean([p['prediction'] for p in predictions.values()])
                ensemble_prob = np.mean([p['probability'] for p in predictions.values()])
                
                results.append({
                    'status': 'success',
                    'drive_id': record.get('Drive_ID', 'unknown'),
                    'predictions': predictions,
                    'ensemble': {
                        'prediction': int(round(ensemble_pred)),
                        'probability': float(ensemble_prob)
                    },
                    'failure_risk': 'HIGH' if ensemble_prob > 0.6 else ('MEDIUM' if ensemble_prob > 0.3 else 'LOW')
                })
            
            except Exception as e:
                results.append({
                    'status': 'error',
                    'drive_id': record.get('Drive_ID', 'unknown'),
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'total_records': len(records),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/statistics', methods=['GET'])
def statistics():
    """Get prediction statistics"""
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE prediction = 1')
        failure_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT model_name, COUNT(*) as count FROM predictions GROUP BY model_name')
        model_stats = cursor.fetchall()
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE prediction = 1 AND model_name = ?',
                      ('random_forest',))
        rf_failures = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_predictions': total_predictions,
            'failure_predictions': failure_predictions,
            'failure_rate': failure_predictions / total_predictions if total_predictions > 0 else 0,
            'model_statistics': [
                {'model': model, 'predictions': count}
                for model, count in model_stats
            ]
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/feature-importance/<model_name>', methods=['GET'])
def feature_importance(model_name):
    """Get feature importance for tree-based models"""
    
    try:
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        model = models[model_name]
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            return jsonify({'error': 'Model does not have feature importances'}), 400
        
        importances = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        feature_importance_dict = {
            feature_names[i]: float(importances[i])
            for i in indices
        }
        
        return jsonify({
            'model': model_name,
            'feature_importance': feature_importance_dict
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/feature-names', methods=['GET'])
def get_feature_names():
    """Get required feature names"""
    return jsonify({
        'features': feature_names,
        'count': len(feature_names)
    })


if __name__ == '__main__':
    print("Initializing backend...")
    init_db()
    load_models()
    load_preprocessing_tools()
    
    print("\nStarting Flask server...")
    print("Visit http://localhost:5001 for health check")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
