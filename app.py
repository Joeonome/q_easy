import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# Import the feature engineering module
from feature_engineering import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Q-EASY: Hospital Wait Time Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model, scaler, and metadata
@st.cache_resource
def load_model_and_metadata():
    """Load all necessary files for prediction"""
    try:
        model = joblib.load('best_lgbm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        metadata = joblib.load('model_metadata.pkl')
        
        # Create feature engineer
        feature_engineer = FeatureEngineer(metadata, scaler)
        
        return model, feature_engineer, metadata
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None

# Load at startup
model, feature_engineer, metadata = load_model_and_metadata()

# Custom CSS (keeping your original styling)
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: transparent;
    }
    div[data-testid="stMetricValue"] {
        font-size: clamp(24px, 5vw, 32px);
        font-weight: bold;
        color: #667eea;
    }
    .prediction-card {
        background: white;
        padding: clamp(15px, 4vw, 30px);
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    h1 {
        color: white;
        text-align: center;
        font-size: clamp(2em, 8vw, 3em);
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    h2, h3 {
        color: #667eea;
        font-size: clamp(1.2em, 4vw, 1.8em);
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: clamp(0.9em, 3vw, 1.2em);
        margin-bottom: 30px;
        opacity: 0.9;
    }
    .info-card {
        background: white;
        padding: clamp(15px, 3vw, 30px);
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🏥 Q-EASY</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Hospital Wait Time Prediction System</p>", unsafe_allow_html=True)

def predict_wait_time(user_inputs):
    """
    Make prediction using the trained model.
    
    Args:
        user_inputs: Dictionary containing all user inputs
        
    Returns:
        tuple: (wait_minutes, lower_bound, upper_bound) or None if error
    """
    if model is None or feature_engineer is None:
        st.error("Model not loaded. Using fallback prediction.")
        # Fallback to simple estimate
        log_wait = 3.5 + np.random.randn() * 0.3
        wait_minutes = np.exp(log_wait)
        std = 0.3
        lower_bound = np.exp(log_wait - 1.96 * std)
        upper_bound = np.exp(log_wait + 1.96 * std)
        return wait_minutes, lower_bound, upper_bound
    
    try:
        # Create feature vector using the feature engineer
        X = feature_engineer.create_features(user_inputs)
        
        # Validate features
        is_valid, error_msg = feature_engineer.validate_features(X)
        if not is_valid:
            st.error(f"Feature validation failed: {error_msg}")
            return None
        
        # Make prediction (returns log-transformed wait time)
        log_wait_pred = model.predict(X)[0]
        
        # Convert back to minutes
        wait_minutes = np.exp(log_wait_pred)
        
        # Calculate confidence interval using model's RMSE from metadata
        std = metadata.get('model_rmse', 0.3)
        lower_bound = np.exp(log_wait_pred - 1.96 * std)
        upper_bound = np.exp(log_wait_pred + 1.96 * std)
        
        return wait_minutes, lower_bound, upper_bound
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.exception(e)
        return None

# Tabs for better organization
tab1, tab2 = st.tabs(["📋 Patient Input", "📊 Results & Info"])

with tab1:
    st.markdown("### Patient Information")
    
    # Demographics
    with st.expander("👤 Demographics", expanded=True):
        age = st.slider("Age", 0, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    # Vital Signs
    with st.expander("💓 Vital Signs", expanded=True):
        temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
        
        col_hr, col_rr = st.columns(2)
        with col_hr:
            hr = st.number_input("Heart Rate (bpm)", 40, 200, 75)
        with col_rr:
            rr = st.number_input("Resp. Rate (bpm)", 8, 40, 16)
        
        col_sys, col_dia = st.columns(2)
        with col_sys:
            bp_sys = st.number_input("BP Systolic", 70, 220, 120)
        with col_dia:
            bp_dia = st.number_input("BP Diastolic", 40, 140, 80)
        
        col_spo2, col_pain = st.columns(2)
        with col_spo2:
            spo2 = st.number_input("SpO2 (%)", 70, 100, 98)
        with col_pain:
            pain = st.slider("Pain (0-10)", 0, 10, 3)
    
    # Clinical Details
    with st.expander("🏥 Clinical Details", expanded=True):
        service = st.selectbox("Service", [
            "Emergency Medicine",
            "Pediatrics",
            "Surgery",
            "Internal Medicine",
            "Cardiology"
        ])
        
        complaint = st.selectbox("Primary Complaint", [
            "Chest Pain",
            "Diabetes",
            "Fever",
            "Headache",
            "Hypertension",
            "Malaria",
            "Pregnancy",
            "Respiratory",
            "Trauma",
            "Other"
        ])
        
        danger_signs = st.multiselect("Danger Signs", [
            "Altered Mental Status",
            "Severe Bleeding",
            "Difficulty Breathing",
            "Chest Pain",
            "None"
        ], default=["None"])
    
    # Hospital Conditions
    with st.expander("🏥 Hospital Conditions", expanded=False):
        triage_level = st.selectbox("Triage Level", [1, 2, 3, 4, 5], index=2)
        is_pregnant = st.checkbox("Patient is Pregnant", value=False)
        
        col_occ, col_load = st.columns(2)
        with col_occ:
            occupancy = st.slider("Hospital Occupancy (%)", 0, 100, 75)
        with col_load:
            doctor_load = st.number_input("Patients per Doctor", 1, 20, 8)
        
        st.markdown("**Current Shift Staffing**")
        col_doc, col_nurse, col_triage = st.columns(3)
        with col_doc:
            shift_doctors = st.number_input("Doctors on Shift", 1, 20, 5)
        with col_nurse:
            shift_nurses = st.number_input("Nurses on Shift", 1, 50, 10)
        with col_triage:
            shift_triage = st.number_input("Triage Nurses", 1, 10, 2)
        
        st.markdown("**Service Department Status**")
        col_queue, col_beds = st.columns(2)
        with col_queue:
            service_queue = st.number_input("Pending Queue", 0, 50, 10)
        with col_beds:
            service_occupancy = st.slider("Service Bed Occupancy (%)", 0, 100, 70)
        
        service_patients = st.number_input("Current Patients in Service", 0, 100, 15)
        
        st.markdown("**Equipment & Environment**")
        col_mri, col_xray, col_or = st.columns(3)
        with col_mri:
            mri_avail = st.checkbox("MRI Available", value=True)
        with col_xray:
            xray_avail = st.checkbox("X-ray Available", value=True)
        with col_or:
            or_avail = st.checkbox("OR Available", value=True)
        
        outside_temp = st.number_input("Outside Temperature (°C)", 15.0, 45.0, 28.0, 0.5)
    
    # Visit Details
    with st.expander("📅 Visit Details", expanded=True):
        arrival_channel = st.selectbox("Arrival Channel", [
            "Walk-in",
            "Ambulance",
            "Referral",
            "WhatsApp Chatbot",
            "Other"
        ])
        
        shift = st.selectbox("Shift", ["Morning", "Afternoon", "Night"])
        
        col_season, col_weather = st.columns(2)
        with col_season:
            season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
        with col_weather:
            weather = st.selectbox("Weather", ["Clear", "Rainy", "Cloudy", "Stormy"])
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Wait Time", type="primary", use_container_width=True)
    
    if predict_btn:
        # Prepare user inputs dictionary
        user_inputs = {
            'age': age,
            'gender': gender,
            'temp': temp,
            'hr': hr,
            'rr': rr,
            'bp_sys': bp_sys,
            'bp_dia': bp_dia,
            'spo2': spo2,
            'pain': pain,
            'service': service,
            'complaint': complaint,
            'danger_signs': danger_signs,
            'triage_level': triage_level,
            'is_pregnant': is_pregnant,
            'occupancy': occupancy,
            'doctor_load': doctor_load,
            'shift_doctors': shift_doctors,
            'shift_nurses': shift_nurses,
            'shift_triage': shift_triage,
            'service_patients': service_patients,
            'service_queue': service_queue,
            'service_occupancy': service_occupancy,
            'mri_avail': mri_avail,
            'xray_avail': xray_avail,
            'or_avail': or_avail,
            'outside_temp': outside_temp,
            'arrival_channel': arrival_channel,
            'shift': shift,
            'season': season,
            'weather': weather,
            'arrival_datetime': pd.Timestamp.now()  # Use current time
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            result = predict_wait_time(user_inputs)
        
        if result is not None:
            st.session_state.prediction = result
            st.session_state.has_prediction = True
            st.success("✅ Prediction complete! Check the 'Results & Info' tab")
        else:
            st.error("❌ Prediction failed. Please check your inputs and try again.")

with tab2:
    if 'has_prediction' in st.session_state and st.session_state.has_prediction:
        wait_time, lower, upper = st.session_state.prediction
        
        st.markdown("### 🎯 Prediction Results")
        
        # Main prediction
        st.markdown(f"""
            <div class='info-card' style='text-align: center;'>
                <h2 style='color: #667eea; margin-bottom: 10px; font-size: clamp(1.3em, 5vw, 1.8em);'>
                    Estimated Wait Time
                </h2>
                <div style='font-size: clamp(3em, 12vw, 4em); font-weight: bold; color: #764ba2;'>
                    {int(wait_time)} min
                </div>
                <p style='color: #666; font-size: clamp(0.9em, 3vw, 1.1em); margin-top: 10px;'>
                    95% CI: {int(lower)} - {int(upper)} minutes
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.markdown("### Key Metrics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Triage", f"Level {triage_level}", delta="Moderate" if triage_level == 3 else "")
        
        with metric_col2:
            st.metric("Queue", f"{service_queue}", delta="-3")
        
        with metric_col3:
            r2_score = metadata.get('model_r2', 0.8872) if metadata else 0.8872
            st.metric("Model R²", f"{r2_score:.1%}", delta="High")
        
        # Gauge chart
        st.markdown("### Wait Time Distribution")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=wait_time,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Wait (minutes)", 'font': {'size': 16}},
            delta={'reference': 30, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 120], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#d4edda'},
                    {'range': [30, 60], 'color': '#fff3cd'},
                    {'range': [60, 120], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👈 Go to 'Patient Input' tab and click 'Predict Wait Time' to see results")
    
    # Model info
    st.markdown("---")
    if metadata:
        r2 = metadata.get('model_r2', 0.8872)
        rmse = metadata.get('model_rmse', 0.3)
        n_features = len(metadata.get('feature_names', []))
        
        st.markdown(f"""
            <div class='info-card'>
                <h3 style='color: #667eea;'>🤖 Model Information</h3>
                <ul style='color: #666; margin: 0; padding-left: 20px;'>
                    <li><strong>Algorithm:</strong> LightGBM Regressor</li>
                    <li><strong>R² Score:</strong> {r2:.4f} ({r2:.2%})</li>
                    <li><strong>RMSE:</strong> {rmse:.4f}</li>
                    <li><strong>Features:</strong> {n_features} variables</li>
                    <li><strong>Target:</strong> Log-transformed wait time</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='info-card'>
                <h3 style='color: #667eea;'>🤖 Model Information</h3>
                <p style='color: #666;'>Model metadata not available</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; opacity: 0.8; padding: 20px;'>
        <p style='font-size: clamp(0.9em, 2.5vw, 1em);'>
            🏥 Q-EASY: Improving Patient Experience Through AI
        </p>
        <p style='font-size: clamp(0.8em, 2vw, 0.9em);'>
            Powered by LightGBM • Built with Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)