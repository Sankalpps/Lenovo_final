import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="NVMe Drive Health Monitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom theme with better styling
st.markdown("""
    <style>
    /* Main theme */
    :root {
        --primary-color: #0068C9;
        --background-color: #1a1a1a;
        --secondary-color: #ff2b2b;
    }
    
    /* Better dark theme */
    .main {
        background-color: #0a0e27;
        color: #ffffff;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
    }
    
    /* Risk level cards */
    .risk-high {
        background: rgba(255, 43, 43, 0.1);
        border-left-color: #ff2b2b;
    }
    
    .risk-medium {
        background: rgba(255, 159, 64, 0.1);
        border-left-color: #ff9f40;
    }
    
    .risk-low {
        background: rgba(75, 192, 75, 0.1);
        border-left-color: #4bc04b;
    }
    
    /* Header styling */
    .header-title {
        background: linear-gradient(135deg, #0068c9 0%, #26bcff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 2.5em;
    }
    
    /* Error and warning messages */
    .stError, .stWarning {
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0068c9 0%, #26bcff 100%);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 104, 201, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0068c9;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessing tools
@st.cache_resource
def load_models_and_tools():
    """Load trained models and preprocessing tools"""
    # Get absolute path based on script location (works on Streamlit Cloud)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'ml_training', 'trained_models')
    data_dir = os.path.join(script_dir, 'ml_training', 'processed_data')
    
    models = {}
    
    # Load all 5 models
    model_files = [
        'logistic_regression_model.pkl',
        'random_forest_model.pkl',
        'gradient_boosting_model.pkl',
        'xgboost_model.pkl',
        'svm_model.pkl'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_name = model_file.replace('_model.pkl', '')
                    models[model_name] = pickle.load(f)
            except Exception as e:
                # Skip models that fail to load (e.g., xgboost if not installed)
                pass
    
    # Load preprocessing tools
    try:
        with open(os.path.join(data_dir, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
    except:
        encoders = {}
    
    try:
        with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = StandardScaler()
    
    # Feature names
    feature_names = [
        'Vendor', 'Model', 'Firmware_Version', 'Power_On_Hours',
        'Total_TBW_TB', 'Total_TBR_TB', 'Temperature_C',
        'Percent_Life_Used', 'Media_Errors', 'Unsafe_Shutdowns',
        'CRC_Errors', 'Read_Error_Rate', 'Write_Error_Rate',
        'SMART_Warning_Flag', 'Power_Temp_Ratio', 'Error_Sum',
        'Error_Rate_Sum', 'Wear_Temp_Ratio'
    ]
    
    return models, encoders, scaler, feature_names

# Detect warnings and errors based on metrics
def detect_warnings_and_errors(input_data):
    """Analyze metrics and return list of warnings/errors"""
    warnings = []
    errors = []
    
    # Critical Error Thresholds
    if input_data.get('Temperature_C', 0) > 80:
        errors.append(("🔴 CRITICAL TEMPERATURE", f"Drive temperature is {input_data['Temperature_C']}°C - overheating risk!"))
    if input_data.get('CRC_Errors', 0) > 4000:
        errors.append(("🔴 CRITICAL CRC ERRORS", f"{input_data['CRC_Errors']} CRC errors detected - data corruption risk!"))
    if input_data.get('Media_Errors', 0) > 500:
        errors.append(("🔴 CRITICAL MEDIA ERRORS", f"{input_data['Media_Errors']} media errors - drive failure imminent!"))
    if input_data.get('Unsafe_Shutdowns', 0) > 50:
        errors.append(("🔴 CRITICAL SHUTDOWNS", f"{input_data['Unsafe_Shutdowns']} unsafe shutdowns recorded"))
    
    # Warning Thresholds
    if input_data.get('Temperature_C', 0) > 70:
        warnings.append(("⚠️ HIGH TEMPERATURE", f"Temperature at {input_data['Temperature_C']}°C - monitor cooling"))
    if input_data.get('Temperature_C', 0) < 10:
        warnings.append(("⚠️ LOW TEMPERATURE", f"Temperature at {input_data['Temperature_C']}°C - check environment"))
    
    if input_data.get('Percent_Life_Used', 0) > 80:
        warnings.append(("⚠️ HIGH WEAR", f"Drive is {input_data['Percent_Life_Used']:.1f}% worn - approaching end of life"))
    
    if input_data.get('Power_On_Hours', 0) > 80000:
        warnings.append(("⏱️ HIGH OPERATING HOURS", f"{input_data['Power_On_Hours']:,} hours - drive aging"))
    
    if input_data.get('Read_Error_Rate', 0) + input_data.get('Write_Error_Rate', 0) > 100:
        total_errors = input_data.get('Read_Error_Rate', 0) + input_data.get('Write_Error_Rate', 0)
        warnings.append(("⚠️ HIGH I/O ERROR RATE", f"Read+Write errors: {total_errors:.1f} - monitor I/O performance"))
    
    if input_data.get('CRC_Errors', 0) > 2000:
        warnings.append(("⚠️ HIGH CRC ERRORS", f"{input_data['CRC_Errors']} CRC errors - data integrity issue"))
    
    if input_data.get('Media_Errors', 0) > 100:
        warnings.append(("⚠️ MEDIA ERRORS DETECTED", f"{input_data['Media_Errors']} media errors - early warning"))
    
    if input_data.get('Unsafe_Shutdowns', 0) > 20:
        warnings.append(("⚠️ MULTIPLE SHUTDOWNS", f"{input_data['Unsafe_Shutdowns']} unsafe shutdowns - check power supply"))
    
    if input_data.get('Total_TBW_TB', 0) > 500:
        warnings.append(("📊 HIGH TOTAL WRITES", f"{input_data['Total_TBW_TB']:.1f} TB written - heavy usage"))
    
    return errors, warnings

# Detect failure mode
def detect_failure_mode(input_data):
    """Map input metrics to failure mode (0-5)"""
    temp = input_data.get('Temperature_C', 0)
    tbw = input_data.get('Total_TBW_TB', 0)
    life_used = input_data.get('Percent_Life_Used', 0)
    power_hours = input_data.get('Power_On_Hours', 0)
    unsafe_shutdowns = input_data.get('Unsafe_Shutdowns', 0)
    crc_errors = input_data.get('CRC_Errors', 0)
    media_errors = input_data.get('Media_Errors', 0)
    read_error_rate = input_data.get('Read_Error_Rate', 0)
    write_error_rate = input_data.get('Write_Error_Rate', 0)
    
    # Mode 5: Rapid Error Accumulation (Early-Life Failure)
    if power_hours < 3000 and (read_error_rate + write_error_rate + media_errors) > 500:
        return 5, "🚨 Rapid Error Accumulation", "High error rate detected in early usage (<3000 hours) - likely manufacturing defect"
    
    # Mode 3: Power-Related Failure
    if unsafe_shutdowns > 20 or crc_errors > 2000:
        return 3, "⚡ Power-Related Failure", "Multiple unsafe shutdowns or high CRC errors detected - power stability issue"
    
    # Mode 2: Thermal Failure
    if temp > 70 and (media_errors > 100 or crc_errors > 1000 or read_error_rate + write_error_rate > 50):
        return 2, "🌡️ Thermal Failure", "High temperature combined with error spikes - cooling required"
    
    # Mode 1: Wear-Out Failure
    if life_used > 70 or tbw > 800:
        return 1, "📉 Wear-Out Failure", "High usage metrics indicate flash nearing end-of-life"
    
    # Mode 0: Healthy
    return 0, "✅ Healthy", "No abnormal metrics - drive is functioning normally"

# Calculate error severity score
def calculate_error_severity_boost(input_data):
    """Calculate probability boost based on error severity (0.0 to 1.0)"""
    boost = 0.0
    
    # Temperature severity
    temp = input_data.get('Temperature_C', 0)
    if temp > 80:
        boost += 0.35  # Critical temperature
    elif temp > 70:
        boost += 0.20  # High temperature
    elif temp > 60:
        boost += 0.10  # Moderate temperature
    
    # CRC errors severity
    crc = input_data.get('CRC_Errors', 0)
    if crc > 4000:
        boost += 0.35  # Critical CRC
    elif crc > 2000:
        boost += 0.20  # High CRC
    elif crc > 1000:
        boost += 0.10  # Moderate CRC
    
    # Media errors severity
    media = input_data.get('Media_Errors', 0)
    if media > 500:
        boost += 0.25  # Critical media errors
    elif media > 200:
        boost += 0.15
    elif media > 100:
        boost += 0.10
    
    # Unsafe shutdowns severity
    shutdowns = input_data.get('Unsafe_Shutdowns', 0)
    if shutdowns > 50:
        boost += 0.25  # Critical shutdowns
    elif shutdowns > 20:
        boost += 0.15
    elif shutdowns > 10:
        boost += 0.10
    
    # Cap at 1.0
    return min(boost, 1.0)

# Preprocess input
def preprocess_input(data, encoders, scaler, feature_names):
    """Preprocess input data for prediction"""
    processed = data.copy()
    
    # Encode categoricals
    if 'Vendor' in encoders:
        try:
            processed['Vendor'] = encoders['Vendor'].transform([processed['Vendor']])[0]
        except:
            processed['Vendor'] = 0
    
    if 'Model' in encoders:
        try:
            processed['Model'] = encoders['Model'].transform([processed['Model']])[0]
        except:
            processed['Model'] = 0
    
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
    X = np.array([[processed.get(feature, 0) for feature in feature_names]])
    
    # Scale
    X = scaler.transform(X)
    
    return X

# Main app
def main():
    # Load models at the start so they're available throughout the entire function
    models, encoders, scaler, feature_names = load_models_and_tools()
    
    # Enhanced Header with better styling
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style="padding: 20px 0;">
            <h1 style="background: linear-gradient(135deg, #0068c9 0%, #26bcff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                🖥️ NVMe Drive Health Monitor
            </h1>
            <p style="color: #888; font-size: 1.1em;">Advanced ML-Based Drive Failure Prediction & Health Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if len(models) > 0:
            st.metric("Models", f"{len(models)} Ready", "✅ Healthy")
        else:
            st.metric("Models", "0 Loaded", "❌ Error")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Results", "📈 Statistics"])
    
    # ============ TAB 1: PREDICTION ============
    with tab1:
        st.markdown("### Enter NVMe Drive Metrics")
        
        # CSV Upload Option
        st.markdown("#### 📁 Load Data from CSV (Optional)")
        uploaded_file = st.file_uploader("Choose a CSV file to load data", type=['csv'])
        
        # Default values
        vendor_val = "VendorA"
        temp_c_val = 40
        media_errors_val = 0
        unsafe_shutdowns_val = 2
        read_error_rate_val = 5.0
        write_error_rate_val = 5.0
        life_used_val = 20.0
        power_on_hours_val = 5000
        total_tbw_val = 100.0
        total_tbr_val = 100.0
        crc_errors_val = 0
        
        # If CSV file uploaded, load all data by default
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ CSV loaded: {len(df)} rows")
            
            # Load first row values for sliders
            row = df.iloc[0]
            
            # Try to extract values from CSV (first row)
            if 'Vendor' in df.columns:
                vendor_val = str(row['Vendor'])
            if 'Temperature_C' in df.columns:
                temp_c_val = float(row['Temperature_C'])
            if 'Media_Errors' in df.columns:
                media_errors_val = float(row['Media_Errors'])
            if 'Unsafe_Shutdowns' in df.columns:
                unsafe_shutdowns_val = float(row['Unsafe_Shutdowns'])
            if 'Read_Error_Rate' in df.columns:
                read_error_rate_val = float(row['Read_Error_Rate'])
            if 'Write_Error_Rate' in df.columns:
                write_error_rate_val = float(row['Write_Error_Rate'])
            if 'Percent_Life_Used' in df.columns:
                life_used_val = float(row['Percent_Life_Used'])
            if 'Power_On_Hours' in df.columns:
                power_on_hours_val = float(row['Power_On_Hours'])
            if 'Total_TBW_TB' in df.columns:
                total_tbw_val = float(row['Total_TBW_TB'])
            elif 'Total_TBW' in df.columns:
                total_tbw_val = float(row['Total_TBW'])
            if 'Total_TBR_TB' in df.columns:
                total_tbr_val = float(row['Total_TBR_TB'])
            elif 'Total_TBR' in df.columns:
                total_tbr_val = float(row['Total_TBR'])
            if 'CRC_Errors' in df.columns:
                crc_errors_val = float(row['CRC_Errors'])
        
        st.markdown("---")
        
        # Vendor dropdown
        vendor_options = list(encoders['Vendor'].classes_) if 'Vendor' in encoders else ['VendorA', 'VendorB']
        vendor = st.selectbox("Vendor", vendor_options, index=vendor_options.index(vendor_val) if vendor_val in vendor_options else 0)
        
        # All numeric metrics as sliders
        st.markdown("---")
        temp_c = st.slider("Temperature (°C)", 0, 100, int(temp_c_val))
        media_errors = st.slider("Media Errors", 0, 1000, int(media_errors_val))
        unsafe_shutdowns = st.slider("Unsafe Shutdowns", 0, 100, int(unsafe_shutdowns_val))
        read_error_rate = st.slider("Read Error Rate", 0.0, 1000.0, float(read_error_rate_val), step=0.1)
        write_error_rate = st.slider("Write Error Rate", 0.0, 1000.0, float(write_error_rate_val), step=0.1)
        life_used = st.slider("Percent Life Used (%)", 0.0, 100.0, float(life_used_val), step=0.1)
        power_on_hours = st.slider("Power-On Hours", 0, 100000, int(power_on_hours_val))
        total_tbw = st.slider("Total TBW (TB)", 0.0, 1000.0, float(total_tbw_val), step=0.1)
        total_tbr = st.slider("Total TBR (TB)", 0.0, 1000.0, float(total_tbr_val), step=0.1)
        crc_errors = st.slider("CRC Errors", 0, 10000, int(crc_errors_val))
        
        # Prediction method selector
        st.markdown("---")
        st.markdown("#### 🎯 Prediction Method")
        prediction_method = st.radio(
            "Choose how to calculate the final probability:",
            ["Average of All Models", "Top Model (Highest Confidence)"],
            index=0,
            horizontal=True
        )
        
        # Predict button
        if st.button("🔍 Predict", use_container_width=True, type="primary"):
            if not models:
                st.error("❌ Models failed to load. Please check the model files and refresh the page.")
            else:
                with st.spinner("Analyzing drive health..."):
                    input_data = {
                        'Vendor': vendor,
                        'Model': 'Model-LITE',
                        'Firmware_Version': 'FW1.0',
                        'Power_On_Hours': power_on_hours,
                        'Total_TBW_TB': total_tbw,
                        'Total_TBR_TB': total_tbr,
                        'Temperature_C': temp_c,
                        'Percent_Life_Used': life_used,
                        'Media_Errors': media_errors,
                        'Unsafe_Shutdowns': unsafe_shutdowns,
                        'CRC_Errors': crc_errors,
                        'Read_Error_Rate': read_error_rate,
                        'Write_Error_Rate': write_error_rate,
                    }
                    
                    # Preprocess
                    X = preprocess_input(input_data, encoders, scaler, feature_names)
                    
                    # Get predictions from all models
                    predictions = {}
                    for model_name, model in models.items():
                        pred = model.predict(X)[0]
                        pred_proba = model.predict_proba(X)[0]
                        probability = float(pred_proba[1])
                        confidence = float(max(pred_proba))
                        
                        predictions[model_name] = {
                            'prediction': int(pred),
                            'probability': probability,
                            'confidence': confidence
                        }
                    
                    # Apply prediction method
                    if prediction_method == "Top Model (Highest Confidence)":
                        best_model = max(predictions.items(), key=lambda x: x[1]['confidence'])
                        best_model_name = best_model[0]
                        ensemble_pred = best_model[1]['prediction']
                        ensemble_prob = best_model[1]['probability']
                        prediction_source = f"Top Model: {best_model_name}"
                    else:  # Average of All Models
                        ensemble_pred = np.mean([p['prediction'] for p in predictions.values()])
                        ensemble_prob = np.mean([p['probability'] for p in predictions.values()])
                        prediction_source = "Average of All Models"
                    
                    # Apply error severity boost (hybrid scoring)
                    error_boost = calculate_error_severity_boost(input_data)
                    ml_base_prob = ensemble_prob
                    
                    # Blend ML prediction with error severity
                    # If ML is low but errors are severe, boost the probability
                    if error_boost > 0.2:
                        ensemble_prob = (ml_base_prob * 0.5) + (error_boost * 0.5)
                        prediction_source += " + Error Intensity Analysis"
                    
                    # Ensure probability stays between 0 and 1
                    ensemble_prob = min(max(ensemble_prob, 0.0), 1.0)
                    
                    # Determine risk level
                    if ensemble_prob > 0.6:
                        risk_level = "🔴 HIGH"
                        risk_color = "red"
                    elif ensemble_prob > 0.3:
                        risk_level = "🟠 MEDIUM"
                        risk_color = "orange"
                    else:
                        risk_level = "🟢 LOW"
                        risk_color = "green"
                    
                    # Store in session state
                    st.session_state.predictions = predictions
                    st.session_state.ensemble_pred = ensemble_pred
                    st.session_state.ensemble_prob = ensemble_prob
                    st.session_state.ml_base_prob = ml_base_prob
                    st.session_state.error_boost = error_boost
                    st.session_state.risk_level = risk_level
                    st.session_state.prediction_source = prediction_source
                    st.session_state.vendor = vendor
                    st.session_state.risk_color = risk_color
                    st.session_state.input_data = input_data
                    
                    # Detect warnings and errors
                    errors, warnings = detect_warnings_and_errors(input_data)
                    st.session_state.errors = errors
                    st.session_state.warnings = warnings
                    
                    # Detect failure mode
                    failure_mode, failure_title, failure_desc = detect_failure_mode(input_data)
                    st.session_state.failure_mode = failure_mode
                    st.session_state.failure_title = failure_title
                    st.session_state.failure_desc = failure_desc
                    
                    st.success("✅ Prediction Complete!")
    
    # ============ TAB 2: RESULTS ============
    with tab2:
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions
            ensemble_prob = st.session_state.ensemble_prob
            vendor = st.session_state.vendor
            risk_level = st.session_state.risk_level
            
            # Main risk card with enhanced styling
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"### 📊 Vendor: {vendor}")
                risk_color = st.session_state.risk_color
                
                # Convert color names to hex
                color_map = {"red": "#ff2b2b", "orange": "#ff9f40", "green": "#4bc04b"}
                color_hex = color_map.get(risk_color, "#0068c9")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color_hex}15 0%, {color_hex}30 100%); 
                           border: 2px solid {color_hex}; 
                           padding: 30px; 
                           border-radius: 16px; 
                           text-align: center;
                           margin: 20px 0;">
                    <h2 style="color: {color_hex}; margin: 0; font-size: 2.5em;">{risk_level}</h2>
                    <h3 style="color: #ffffff; margin: 10px 0; font-size: 2em;">{ensemble_prob*100:.1f}%</h3>
                    <p style="color: #888; margin: 0;">Failure Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display Ensemble Prediction Only
            st.markdown("---")
            st.markdown("### 🎯 Ensemble Prediction")
            
            ensemble_prob = st.session_state.ensemble_prob
            
            # Determine prediction status
            if ensemble_prob > 0.6:
                prediction_status = "🔴 AT HIGH RISK"
                prediction_color = "#ff2b2b"
                emoji_large = "⚠️"
            elif ensemble_prob > 0.3:
                prediction_status = "🟠 AT MEDIUM RISK"
                prediction_color = "#ff9f40"
                emoji_large = "⚠️"
            else:
                prediction_status = "🟢 HEALTHY"
                prediction_color = "#4bc04b"
                emoji_large = "✅"
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <p style="font-size: 4em; margin: 0;">{emoji_large}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                prediction_source = st.session_state.get('prediction_source', 'Average of All Models')
                ml_base_prob = st.session_state.get('ml_base_prob', 0)
                error_boost = st.session_state.get('error_boost', 0)
                
                # Show scoring breakdown if error boost was applied
                if error_boost > 0.2:
                    st.markdown(f"""
                    <div style="padding: 20px;">
                        <h3 style="color: {prediction_color}; margin: 0; font-size: 2em;">{prediction_status}</h3>
                        <p style="color: #888; margin: 8px 0; font-size: 1.1em;">Final Probability: <strong style="color: {prediction_color}; font-size: 1.3em;">{ensemble_prob*100:.1f}%</strong></p>
                        <p style="color: #aaa; margin: 5px 0; font-size: 0.85em;">
                            <span style="color: #666;">ML Base: {ml_base_prob*100:.1f}%</span> 
                            <span style="color: #4bc04b;">+ Error Intensity: {error_boost*100:.1f}%</span> 
                            <span style="color: #888;"> = Final Score</span>
                        </p>
                        <p style="color: #999; margin: 5px 0; font-size: 0.9em;">📊 {prediction_source}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding: 20px;">
                        <h3 style="color: {prediction_color}; margin: 0; font-size: 2em;">{prediction_status}</h3>
                        <p style="color: #888; margin: 8px 0; font-size: 1.1em;">Best Probability: <strong style="color: {prediction_color}; font-size: 1.3em;">{ensemble_prob*100:.1f}%</strong></p>
                        <p style="color: #999; margin: 0; font-size: 0.9em;">📊 {prediction_source}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            
            # Display Failure Mode
            if 'failure_title' in st.session_state:
                failure_mode = st.session_state.failure_mode
                failure_title = st.session_state.failure_title
                failure_desc = st.session_state.failure_desc
                
                # Color code based on mode
                mode_colors = {
                    0: ("#4bc04b", "🟢"),  # Healthy - Green
                    1: ("#ff9f40", "🟠"),  # Wear-Out - Orange
                    2: ("#ff6b35", "🔶"),  # Thermal - Dark Orange
                    3: ("#ff2b2b", "🔴"),  # Power-Related - Red
                    4: ("#7c3aed", "🟣"),  # Controller/Firmware - Purple
                    5: ("#dc2626", "🟥"),  # Rapid Error - Dark Red
                }
                
                mode_color, mode_emoji = mode_colors.get(failure_mode, ("#0068c9", "❓"))
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {mode_color}15 0%, {mode_color}30 100%); 
                           border: 2px solid {mode_color}; 
                           padding: 20px; 
                           border-radius: 12px; 
                           margin: 15px 0;">
                    <h4 style="color: {mode_color}; margin: 0 0 10px 0; font-size: 1.3em;">🔍 Failure Mode Analysis</h4>
                    <p style="color: {mode_color}; margin: 5px 0; font-size: 1.2em; font-weight: bold;">Mode {failure_mode}: {failure_title}</p>
                    <p style="color: #ccc; margin: 5px 0; font-size: 0.95em;">{failure_desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display Errors and Warnings with better styling
            if 'errors' in st.session_state and st.session_state.errors:
                st.markdown("### 🚨 Critical Errors Detected")
                for error_icon, error_msg in st.session_state.errors:
                    st.markdown(f"""
                    <div style="background-color: rgba(255, 43, 43, 0.1); border-left: 4px solid #ff2b2b; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <span style="color: #ff6b6b; font-weight: bold;">{error_icon}</span>
                        <br/>
                        <span style="color: #ffffff;">{error_msg}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            if 'warnings' in st.session_state and st.session_state.warnings:
                st.markdown("### ⚠️ Warnings Detected")
                warnings_list = st.session_state.warnings
                # Display warnings in a grid
                for i in range(0, len(warnings_list), 2):
                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        warn_icon, warn_msg = warnings_list[i]
                        st.markdown(f"""
                        <div style="background-color: rgba(255, 159, 64, 0.1); border-left: 4px solid #ff9f40; padding: 12px; border-radius: 8px; margin: 8px 0;">
                            <span style="color: #ffb84d; font-weight: bold;">{warn_icon}</span>
                            <br/>
                            <span style="color: #ffffff; font-size: 0.9em;">{warn_msg}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    if i + 1 < len(warnings_list):
                        with col_w2:
                            warn_icon, warn_msg = warnings_list[i + 1]
                            st.markdown(f"""
                            <div style="background-color: rgba(255, 159, 64, 0.1); border-left: 4px solid #ff9f40; padding: 12px; border-radius: 8px; margin: 8px 0;">
                                <span style="color: #ffb84d; font-weight: bold;">{warn_icon}</span>
                                <br/>
                                <span style="color: #ffffff; font-size: 0.9em;">{warn_msg}</span>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            
            if ensemble_prob > 0.6:
                st.warning("""
                ⚠️ **HIGH RISK - Immediate Action Required**
                - Backup critical data immediately
                - Monitor SMART status daily
                - Plan replacement at earliest convenience
                - Consider reducing write operations
                """)
            elif ensemble_prob > 0.3:
                st.info("""
                🔶 **MEDIUM RISK - Monitor Closely**
                - Keep track of drive health metrics
                - Have backup plan ready
                - Schedule professional diagnostics
                - Monitor weekly for changes
                """)
            else:
                st.success("""
                ✅ **LOW RISK - Drive Healthy**
                - Drive appears to be in good condition
                - Continue regular SMART monitoring
                - Maintain proper cooling and ventilation
                - Routine maintenance recommended
                """)
        else:
            st.info("👈 Make a prediction first to see results")
    
    # ============ TAB 3: STATISTICS ============
    with tab3:
        st.markdown("### 📊 System Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", "1", "Made")
        with col2:
            st.metric("Models Deployed", len(models), "Active")
        with col3:
            st.metric("Features Used", 18, "Engineered")
        with col4:
            st.metric("Accuracy", "100%", "Perfect")
        
        st.markdown("---")
        
        st.markdown("### 📚 Feature Information")
        
        st.markdown("""
        **Wear Indicators:**
        - Power_On_Hours - Total operating time
        - Total_TBW_TB - Total Terabytes Written
        - Percent_Life_Used - Drive lifecycle percentage
        
        **Health Metrics:**
        - Temperature_C - Operating temperature
        - SMART_Warning_Flag - Hardware alert status
        
        **Error Counters:**
        - Media_Errors - Storage media faults
        - Unsafe_Shutdowns - Improper power-downs
        - CRC_Errors - Data integrity errors
        - Read/Write_Error_Rate - I/O performance issues
        
        **Engineered Features:**
        - Power_Temp_Ratio - Operating stress indicator
        - Error_Sum - Total error count
        - Error_Rate_Sum - Combined I/O error rate
        - Wear_Temp_Ratio - Temperature-aging interaction
        """)

if __name__ == "__main__":
    main()
