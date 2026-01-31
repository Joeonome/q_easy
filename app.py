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


LOGO_URL = "https://i.ibb.co/1J2sSj92/q-easy-streamlit-header.png"

# Page configuration
st.set_page_config(
    page_title="Q-EASY: Hospital Wait Time Predictor",
    page_icon=LOGO_URL,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.image(LOGO_URL, width=80)

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

def get_triage_category(triage_level, wait_time, user_inputs):
    """
    Automatically assign triage category based on triage level, 
    predicted wait time, and clinical signals.
    
    Triage Level Interpretation:
    - Level 1: Resuscitation - Immediate life-threatening (shortest wait)
    - Level 2: Emergent - Very urgent (very short wait)
    - Level 3: Urgent - Moderate urgency (moderate wait)
    - Level 4: Less Urgent - Can wait (longer wait)
    - Level 5: Non-Urgent - Minor issues (longest wait)
    
    Args:
        triage_level: Integer from 1-5
        wait_time: Predicted wait time in minutes
        user_inputs: Dictionary of patient inputs
        
    Returns:
        str: 'Emergency', 'Urgent', or 'Routine'
    """
    # Extract relevant clinical signals safely
    danger_signs = user_inputs.get('danger_signs', [])
    has_danger_signs = any(sign != "None" for sign in danger_signs)
    
    pain_level = user_inputs.get('pain', 0)
    spo2 = user_inputs.get('spo2', 100)
    hr = user_inputs.get('hr', 75)
    bp_sys = user_inputs.get('bp_sys', 120)
    temp = user_inputs.get('temp', 37.0)
    
    # Critical vital signs indicators
    critical_vitals = (
        spo2 < 90 or 
        hr > 120 or hr < 50 or 
        bp_sys > 180 or bp_sys < 90 or
        temp > 39.5 or temp < 35.5
    )
    
    # Rule-based categorization
    # Lower triage levels (1-2) are MORE urgent = Emergency/Urgent
    # Higher triage levels (4-5) are LESS urgent = Routine
    if triage_level == 1:
        return "Emergency"
    elif triage_level == 2:
        if has_danger_signs or critical_vitals:
            return "Emergency"
        else:
            return "Urgent"
    elif triage_level == 3:
        if has_danger_signs or critical_vitals or pain_level >= 8:
            return "Urgent"
        else:
            return "Routine"
    elif triage_level == 4:
        if has_danger_signs or pain_level >= 9 or critical_vitals:
            return "Urgent"
        else:
            return "Routine"
    else:  # triage_level == 5
        if has_danger_signs or critical_vitals:
            return "Urgent"
        else:
            return "Routine"

def get_hospital_interventions(wait_time, triage_level, triage_category, user_inputs):
    """
    Generate hospital-specific intervention recommendations.
    
    Args:
        wait_time: Predicted wait time in minutes
        triage_level: Integer from 1-5 (1=most urgent, 5=least urgent)
        triage_category: 'Emergency', 'Urgent', or 'Routine'
        user_inputs: Dictionary of patient inputs
        
    Returns:
        list: List of intervention strings
    """
    interventions = []
    
    # Extract relevant metrics safely with defaults
    service_queue = user_inputs.get('service_queue', 0)
    occupancy = user_inputs.get('occupancy', 0)
    doctor_load = user_inputs.get('doctor_load', 0)
    shift_doctors = user_inputs.get('shift_doctors', 5)
    shift_nurses = user_inputs.get('shift_nurses', 10)
    shift_triage = user_inputs.get('shift_triage', 2)
    service_occupancy = user_inputs.get('service_occupancy', 0)
    
    # Emergency-specific interventions (Triage 1-2)
    if triage_category == "Emergency" or triage_level <= 2:
        interventions.append("🚨 PRIORITY: Activate emergency protocol - immediate physician assessment required")
        interventions.append("💉 Prepare emergency bay and assign dedicated nurse immediately")
        if wait_time > 15:
            interventions.append("⚡ CRITICAL: Patient wait exceeds emergency threshold - fast-track immediately and bypass standard queue")
        if triage_level == 1:
            interventions.append("🏥 Level 1 Resuscitation - Assign to resuscitation area with full trauma team on standby")
    
    # Triage level 1-2 specific interventions
    if triage_level <= 2:
        interventions.append("🏥 Assign to resuscitation area or critical care bay")
        if doctor_load > 6:
            interventions.append("👨‍⚕️ Redistribute patient load - reassign 2-3 stable patients to available physicians")
        if wait_time > 10:
            interventions.append("⏰ URGENT: Triage 1-2 patient waiting >10 minutes - immediate intervention required")
    
    # High wait time interventions
    if wait_time > 90:
        interventions.append("⏰ Critical wait time alert - escalate to shift supervisor immediately")
        if service_queue > 8:
            interventions.append(f"📋 PRIORITY: Prioritize the 3 longest-waiting urgent patients to prevent backlog escalation")
    elif wait_time > 60:
        if service_queue > 5:
            interventions.append("📊 Monitor queue actively - consider opening additional treatment rooms")
        if triage_level <= 3:
            interventions.append("⚠️ Urgent/Emergency patient experiencing extended wait - review and expedite if possible")
    
    # Staffing interventions
    if doctor_load > 8:
        interventions.append(f"👨‍⚕️ High doctor load detected ({doctor_load} patients/doctor) - request additional physician support")
    
    if shift_triage < 2 and service_queue > 10:
        interventions.append("👩‍⚕️ Add 1 extra triage nurse during peak hours (e.g., 10-11 AM) to improve patient flow")
    
    if occupancy > 85:
        interventions.append("🛏️ Hospital near capacity - coordinate with bed management for potential admissions")
        if service_occupancy > 90:
            interventions.append("🔄 Service department at critical occupancy - prepare discharge plan for stable patients")
    
    # Queue management
    if service_queue > 15:
        interventions.append("📢 Implement queue management protocol - provide regular updates to waiting patients")
        interventions.append("🔍 Review queue for patients suitable for discharge with outpatient follow-up")
    
    # Service-specific interventions
    service = user_inputs.get('service', '')
    if service == "Surgery" and wait_time > 45:
        or_avail = user_inputs.get('or_avail', True)
        if not or_avail:
            interventions.append("🏥 No OR available - coordinate with regional hospitals for transfer if needed")
        else:
            interventions.append("🔧 Prepare OR and surgical team for potential emergency procedure")
    
    if service == "Obstetrics & Gynecology":
        is_pregnant = user_inputs.get('is_pregnant', False)
        if is_pregnant:
            interventions.append("🤰 Obstetric patient - notify L&D unit and prepare fetal monitoring equipment")
            if triage_level <= 2:
                interventions.append("⚠️ URGENT: High-risk obstetric case - immediate OB/GYN consultation required")
    
    if service == "Internal Medicine":
        if triage_level <= 2:
            interventions.append("🏥 Urgent ICU care may be needed - alert ICU team and prepare potential admission")
    
    # Diagnostic equipment
    mri_avail = user_inputs.get('mri_avail', True)
    xray_avail = user_inputs.get('xray_avail', True)
    
    if not mri_avail or not xray_avail:
        unavailable = []
        if not mri_avail:
            unavailable.append("MRI")
        if not xray_avail:
            unavailable.append("X-ray")
        interventions.append(f"🔧 {', '.join(unavailable)} currently unavailable - arrange external imaging if critical")
    
    # Low priority cases with long waits (Triage 4-5)
    if triage_level >= 4 and wait_time > 60:
        interventions.append("💡 Non-urgent case with extended wait - consider offering rescheduling or referral to urgent care clinic")
    
    return interventions

def get_patient_guidance(wait_time, triage_level, triage_category, user_inputs):
    """
    Generate patient-specific guidance and expectations.
    
    Args:
        wait_time: Predicted wait time in minutes
        triage_level: Integer from 1-5 (1=most urgent, 5=least urgent)
        triage_category: 'Emergency', 'Urgent', or 'Routine'
        user_inputs: Dictionary of patient inputs
        
    Returns:
        list: List of guidance strings
    """
    guidance = []
    
    pain_level = user_inputs.get('pain', 0)
    danger_signs = user_inputs.get('danger_signs', [])
    has_danger_signs = any(sign != "None" for sign in danger_signs)
    
    # Emergency category guidance (Triage 1-2)
    if triage_category == "Emergency" or triage_level <= 2:
        guidance.append("🚨 You will be seen immediately - this is a medical emergency")
        guidance.append("🛏️ Please proceed to the emergency treatment area as directed by staff")
        guidance.append("👨‍👩‍👧 Family members may be asked to wait in the designated area")
        if triage_level == 1:
            guidance.append("⚕️ You are our highest priority - medical team is being assembled now")
        return guidance  # Return early for emergency cases
    
    # Wait time expectations
    if wait_time < 15:
        guidance.append(f"⏱️ Expected wait time: {int(wait_time)} minutes - You will be seen very soon")
        guidance.append("⚡ High priority case - you're near the front of the queue")
    elif wait_time < 30:
        guidance.append(f"⏱️ Expected wait time: {int(wait_time)} minutes - You should be seen shortly")
    elif wait_time < 60:
        guidance.append(f"⏱️ Expected wait time: {int(wait_time)} minutes - Average wait for your condition")
        guidance.append("☕ Feel free to use the waiting area facilities (restroom, water fountain)")
    else:
        guidance.append(f"⏱️ Expected wait time: {int(wait_time)} minutes - Longer than average due to current demand")
        guidance.append("🍽️ You may have time for a light snack if your condition allows")
        guidance.append("📱 Consider notifying family members of the extended wait")
    
    # Triage-specific guidance
    if triage_level <= 2:
        guidance.append("⚠️ Your condition is urgent - you will be prioritized over routine cases")
        guidance.append("📍 Please stay in the waiting area and inform staff immediately if symptoms worsen")
    elif triage_level == 3:
        guidance.append("📋 You are in the standard priority queue")
        if has_danger_signs:
            guidance.append("⚠️ Alert staff immediately if you experience any danger signs")
    else:  # Triage 4-5
        guidance.append("📋 Your condition is non-urgent - more critical cases will be seen first")
        if triage_level == 5:
            guidance.append("🏠 Consider: If symptoms remain mild, an outpatient appointment or urgent care clinic may be faster")
    
    # Pain management
    if pain_level >= 7:
        guidance.append("💊 Pain medication may be available - please request from triage nurse if needed")
    
    # Symptom monitoring
    if triage_category == "Urgent" or triage_level <= 3:
        guidance.append("👀 Monitor your symptoms - inform staff immediately if you experience:")
        symptoms_to_watch = []
        
        if pain_level >= 5:
            symptoms_to_watch.append("Worsening or severe pain")
        symptoms_to_watch.append("Difficulty breathing or chest pain")
        symptoms_to_watch.append("Loss of consciousness or confusion")
        symptoms_to_watch.append("Heavy bleeding or severe allergic reaction")
        
        for symptom in symptoms_to_watch:
            guidance.append(f"  • {symptom}")
    
    # General comfort
    guidance.append("💺 Make yourself comfortable in the waiting area")
    guidance.append("❓ Don't hesitate to ask staff questions about your wait or condition")
    
    return guidance

# Custom CSS - FIXED: Ensured text is visible with proper contrast
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
    .intervention-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        color: #333;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .emergency-badge {
        background: #dc3545;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 16px;
    }
    .urgent-badge {
        background: #ffc107;
        color: #000;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 16px;
    }
    .routine-badge {
        background: #28a745;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>Q-EASY</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Hospital Wait Time Prediction System</p>", unsafe_allow_html=True)

def predict_wait_time(user_inputs):
    """
    Make prediction using the trained model.
    
    IMPORTANT: The model was trained with REVERSED triage encoding:
    - Model expects: 5=most urgent, 1=least urgent
    - Users provide: 1=most urgent, 5=least urgent
    - Solution: Reverse triage level before prediction (6 - triage_level)
    
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
        # CRITICAL FIX: Reverse triage level for model
        # User input: 1=most urgent, 5=least urgent (medical standard)
        # Model expects: 5=most urgent, 1=least urgent (reversed)
        # Transform: new_triage = 6 - user_triage
        model_inputs = user_inputs.copy()
        original_triage = user_inputs['triage_level']
        model_inputs['triage_level'] = 6 - original_triage
        
        # Create feature vector using the feature engineer
        X = feature_engineer.create_features(model_inputs)
        
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
        std = metadata.get('model_rmse', 0.3) if metadata else 0.3
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
            rr = st.number_input("Resp. Rate (cpm)", 8, 40, 16)
        
        col_sys, col_dia = st.columns(2)
        with col_sys:
            bp_sys = st.number_input("BP Systolic (mmHg)", 70, 240, 120)
        with col_dia:
            bp_dia = st.number_input("BP Diastolic (mmHg)", 40, 140, 80)
        
        col_spo2, col_pain = st.columns(2)
        with col_spo2:
            spo2 = st.number_input("SpO2 (%)", 60, 100, 98)
        with col_pain:
            pain = st.slider("Pain (0-10)", 0, 10, 3)
    
    # Clinical Details
    with st.expander("🏥 Clinical Details", expanded=True):
        service = st.selectbox("Service", [
            "Emergency Medicine",
            "Pediatrics",
            "Surgery",
            "Internal Medicine",
            "Obstetrics & Gynecology",
        ])
        
        complaint = st.selectbox("Primary Complaint", [
            "Chest Pain",
            "Cough",
            "Frequent Urination",
            "Fever",
            "Headache",
            "Blurred Vision",
            "Pregnancy",
            "Difficulty Breathing",
            "Trauma",
            "Other"
        ])
        
        danger_signs = st.multiselect("Danger Signs", [
            "Loss of Consciousness",
            "High Fever",
            "Altered Mental Status",
            "Severe Bleeding",
            "Difficulty Breathing",
            "Severe Chest Pain",
            "None"
        ], default=["None"])
    
    # Hospital Conditions
    with st.expander("🏥 Hospital Conditions", expanded=True):
        triage_level = st.selectbox("Triage Level (1=Most Urgent, 5=Least Urgent)", [1, 2, 3, 4, 5], index=2)
        st.caption("Level 1: Resuscitation | Level 2: Emergent | Level 3: Urgent | Level 4: Less Urgent | Level 5: Non-Urgent")
        
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
            season = st.selectbox("Season", ["Dry Season", "Rainy Season"])
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
        with st.spinner("🔄 Making prediction..."):
            result = predict_wait_time(user_inputs)
        
        if result is not None:
            wait_time, lower, upper = result
            
            # Get triage category
            triage_category = get_triage_category(triage_level, wait_time, user_inputs)
            
            # Store in session state
            st.session_state.prediction = result
            st.session_state.triage_category = triage_category
            st.session_state.user_inputs = user_inputs
            st.session_state.has_prediction = True
            
            st.success("✅ Prediction complete! Check the 'Results & Info' tab")
        else:
            st.error("❌ Prediction failed. Please check your inputs and try again.")

with tab2:
    if 'has_prediction' in st.session_state and st.session_state.has_prediction:
        wait_time, lower, upper = st.session_state.prediction
        triage_category = st.session_state.get('triage_category', 'Routine')
        user_inputs = st.session_state.get('user_inputs', {})
        triage_level = user_inputs.get('triage_level', 3)
        
        # Triage category badge
        badge_class = f"{triage_category.lower()}-badge"
        st.markdown(f"""
            <div style='text-align: center; margin-bottom: 20px;'>
                <span class='{badge_class}'>{triage_category.upper()}</span>
            </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown("### 📊 Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Triage", f"Level {triage_level}", delta=triage_category)
        
        with metric_col2:
            service_queue = user_inputs.get('service_queue', 0)
            st.metric("Queue", f"{service_queue}", delta="-3" if service_queue > 5 else "Low")
        
        with metric_col3:
            occupancy = user_inputs.get('occupancy', 0)
            st.metric("Occupancy", f"{occupancy}%", delta="High" if occupancy > 80 else "Normal")
        
        with metric_col4:
            r2_score = metadata.get('model_r2', 0.8872) if metadata else 0.8872
            st.metric("Model R²", f"{r2_score:.1%}", delta="High")
        
        # Gauge chart
        st.markdown("### ⏱️ Wait Time Distribution")
        
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
        
        # Hospital Interventions
        st.markdown("---")
        st.markdown("### 🏥 Recommended Hospital Interventions")
        
        hospital_interventions = get_hospital_interventions(
            wait_time, triage_level, triage_category, user_inputs
        )
        
        if hospital_interventions:
            for intervention in hospital_interventions:
                st.markdown(f"""
                    <div class='intervention-card'>
                        {intervention}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='intervention-card' style='border-left-color: #28a745;'>
                    ✅ No special interventions needed - standard care protocol applies
                </div>
            """, unsafe_allow_html=True)
        
        # Patient Guidance
        st.markdown("---")
        st.markdown("### 👤 Patient Guidance & Expectations")
        
        patient_guidance = get_patient_guidance(
            wait_time, triage_level, triage_category, user_inputs
        )
        
        if patient_guidance:
            for guide in patient_guidance:
                st.markdown(f"""
                    <div class='intervention-card' style='border-left-color: #28a745;'>
                        {guide}
                    </div>
                """, unsafe_allow_html=True)
        
    else:
        st.info("👈 Go to 'Patient Input' tab and click 'Predict Wait Time' to see results")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; opacity: 0.8; padding: 20px;'>
        <p style='font-size: clamp(0.9em, 2.5vw, 1em);'>
            Q-EASY: Improving Healthcare Experience Through AI
        </p>
        <p style='font-size: clamp(0.8em, 2vw, 0.9em);'>
            Powered by LightGBM • Built with Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)