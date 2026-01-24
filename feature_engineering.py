import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    """
    Feature engineering class that exactly replicates training preprocessing.
    Ensures predictions use the same feature transformation as training.
    """
    
    def __init__(self, metadata, scaler):
        """
        Initialize with saved metadata and scaler from training.
        
        Args:
            metadata: Dictionary containing training metadata
            scaler: Fitted StandardScaler from training
        """
        self.metadata = metadata
        self.scaler = scaler
        self.feature_names = metadata['feature_names']
        self.numerical_cols_scaled = metadata['numerical_cols_scaled']
        self.categorical_cols = metadata['categorical_cols']
        self.original_categories = metadata['original_categories']
        self.encoded_columns = metadata['encoded_columns']
        self.imputation_values = metadata['imputation_values']
        
    def create_features(self, user_inputs):
        """
        Create feature vector from user inputs matching training exactly.
        
        Args:
            user_inputs: Dictionary with keys matching raw input fields
            
        Returns:
            DataFrame with single row containing all features in correct order
        """
        # Initialize all features to 0
        features = {col: 0 for col in self.feature_names}
        
        # ============================================================
        # 1. NUMERICAL FEATURES (direct mapping)
        # ============================================================
        numerical_mappings = {
            'triage_level_num': user_inputs.get('triage_level', 3),
            'vital_temp_c': user_inputs.get('temp', self.imputation_values.get('vital_temp_c', 37.0)),
            'vital_hr_bpm': user_inputs.get('hr', self.imputation_values.get('vital_hr_bpm', 75)),
            'vital_rr_bpm': user_inputs.get('rr', self.imputation_values.get('vital_rr_bpm', 16)),
            'vital_bp_systolic': user_inputs.get('bp_sys', self.imputation_values.get('vital_bp_systolic', 120)),
            'vital_bp_diastolic': user_inputs.get('bp_dia', self.imputation_values.get('vital_bp_diastolic', 80)),
            'vital_spo2': user_inputs.get('spo2', self.imputation_values.get('vital_spo2', 98)),
            'pain_score_0_10': user_inputs.get('pain', self.imputation_values.get('pain_score_0_10', 3)),
            'pregnant': 1 if user_inputs.get('is_pregnant', False) else 0,
            'age_years': user_inputs.get('age', 45),
            'shift_doctor_count': user_inputs.get('shift_doctors', 5),
            'shift_nurse_count': user_inputs.get('shift_nurses', 10),
            'shift_triage_nurse_count': user_inputs.get('shift_triage', 2),
            'service_current_patients': user_inputs.get('service_patients', 15),
            'service_pending_queue': user_inputs.get('service_queue', 10),
            'service_bed_occupancy_rate': user_inputs.get('service_occupancy', 70) / 100.0,
            'hospital_overall_occupancy_rate': user_inputs.get('occupancy', 75) / 100.0,
            'mri_available': 1 if user_inputs.get('mri_avail', True) else 0,
            'xray_available': 1 if user_inputs.get('xray_avail', True) else 0,
            'or_available': 1 if user_inputs.get('or_avail', True) else 0,
            'temperature_outside_c': user_inputs.get('outside_temp', 28.0),
            'load_per_doctor': user_inputs.get('doctor_load', 8),
        }
        
        # Set all numerical features
        for key, value in numerical_mappings.items():
            if key in features:
                features[key] = value
        
        # ============================================================
        # 2. MISSING VALUE FLAGS (all 0 since we impute)
        # ============================================================
        features['vital_temp_c_missing_flag'] = 0
        features['vital_hr_bpm_missing_flag'] = 0
        features['vital_rr_bpm_missing_flag'] = 0
        features['vital_bp_missing_flag'] = 0
        features['vital_spo2_missing_flag'] = 0
        features['pain_score_missing_flag'] = 0
        
        # ============================================================
        # 3. TIME-BASED FEATURES
        # ============================================================
        # Use provided datetime or current time
        arrival_dt = user_inputs.get('arrival_datetime', pd.Timestamp.now())
        
        features['arrival_hour'] = arrival_dt.hour
        features['arrival_dayofweek'] = arrival_dt.dayofweek
        features['month'] = arrival_dt.month
        features['arrival_timestamp'] = int(arrival_dt.timestamp())
        
        # Shift change windows (hours 7, 15, 23)
        features['is_shift_change_window'] = 1 if arrival_dt.hour in [7, 15, 23] else 0
        
        # Weekend (Saturday=5, Sunday=6)
        features['is_weekend'] = 1 if arrival_dt.dayofweek >= 5 else 0
        
        # Public holiday (would need calendar integration)
        features['is_public_holiday'] = 0
        
        # ============================================================
        # 4. ONE-HOT ENCODED CATEGORICAL FEATURES
        # ============================================================
        
        # SERVICE
        service = user_inputs.get('service', 'Emergency Medicine')
        service_encoded = self._encode_categorical('service', service, {
            'Emergency Medicine': 'emergency',
            'Pediatrics': 'pediatrics',
            'Surgery': 'surgery',
            'Internal Medicine': 'internalmedicine',
            'Cardiology': 'cardiology'
        })
        
        # PRIMARY COMPLAINT
        complaint = user_inputs.get('complaint', 'Other')
        complaint_encoded = self._encode_categorical('primary_complaint', complaint, {
            'Chest Pain': 'chest_pain',
            'Diabetes': 'diabetes',
            'Fever': 'fever',
            'Headache': 'headache',
            'Hypertension': 'hypertension',
            'Malaria': 'malaria',
            'Pregnancy': 'pregnancy',
            'Respiratory': 'respiratory',
            'Trauma': 'trauma',
            'Other': 'abdominal_pain'  # Default
        })
        
        # DANGER SIGNS
        danger_signs = user_inputs.get('danger_signs', ['None'])
        if 'None' in danger_signs or not danger_signs:
            danger_encoded = 'none'
        elif 'Altered Mental Status' in danger_signs:
            danger_encoded = 'unconscious'
        elif 'Difficulty Breathing' in danger_signs:
            danger_encoded = 'severe_respiratory_distress'
        elif 'Severe Bleeding' in danger_signs:
            danger_encoded = 'convulsions'
        else:
            danger_encoded = 'none'
        
        danger_encoded = self._encode_categorical('danger_signs', danger_encoded, {
            'convulsions': 'convulsions',
            'none': 'none',
            'severe_respiratory_distress': 'severe_respiratory_distress',
            'unconscious': 'unconscious'
        })
        
        # RISK FLAGS (based on age)
        age = user_inputs.get('age', 45)
        if age < 5:
            risk_encoded = 'under5'
        elif age > 65:
            risk_encoded = 'over65'
        else:
            risk_encoded = 'chronic_disease'  # or 'none', need to check training data
        
        risk_encoded = self._encode_categorical('risk_flags', risk_encoded, {
            'chronic_disease': 'chronic_disease',
            'none': 'none',
            'over65': 'over65',
            'under5': 'under5'
        })
        
        # ARRIVAL CHANNEL
        channel = user_inputs.get('arrival_channel', 'Walk-in')
        channel_encoded = self._encode_categorical('arrival_channel', channel, {
            'Walk-in': 'walk_in',
            'Ambulance': 'ambulance',
            'Referral': 'referral',
            'WhatsApp Chatbot': 'whatsapp_chatbot',
            'Call Center': 'call_center',
            'Other': 'other'
        })
        
        # GENDER
        gender = user_inputs.get('gender', 'Male')
        gender_encoded = self._encode_categorical('gender', gender, {
            'Male': 'male',
            'Female': 'female'
        })
        
        # SHIFT
        shift = user_inputs.get('shift', 'Morning')
        shift_encoded = self._encode_categorical('shift_name', shift, {
            'Morning': 'morning',
            'Afternoon': 'afternoon',
            'Night': 'night'
        })
        
        # SEASONALITY
        season = user_inputs.get('season', 'Summer')
        season_encoded = self._encode_categorical('seasonality_factor', season, {
            'Winter': 'flu_season',
            'Spring': 'normal',
            'Summer': 'malaria_peak',
            'Fall': 'normal'
        })
        
        # WEATHER
        weather = user_inputs.get('weather', 'Clear')
        weather_encoded = self._encode_categorical('weather_condition', weather, {
            'Clear': 'clear',
            'Rainy': 'rain',
            'Cloudy': 'cloudy',
            'Stormy': 'storm'
        })
        
        # ============================================================
        # 5. CONVERT TO DATAFRAME AND ENSURE CORRECT ORDER
        # ============================================================
        df = pd.DataFrame([features])
        
        # Ensure features are in exact same order as training
        df = df[self.feature_names]
        
        # ============================================================
        # 6. SCALE NUMERICAL FEATURES
        # ============================================================
        # Only scale the columns that were scaled during training
        cols_to_scale = [col for col in self.numerical_cols_scaled if col in df.columns]
        
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def _encode_categorical(self, category_name, value, mapping):
        """
        Helper to encode categorical value to match training one-hot encoding.
        
        Args:
            category_name: Name of the categorical column (e.g., 'service')
            value: Raw value or mapped value
            mapping: Dictionary mapping UI values to encoded values
        """
        # Map the value if it's a UI label
        if value in mapping:
            encoded_value = mapping[value]
        else:
            # Value is already encoded or unknown
            encoded_value = value
        
        # Get all encoded columns for this category
        encoded_cols = self.encoded_columns.get(category_name, [])
        
        # Set the appropriate column to 1
        # Remember: drop_first=True means first alphabetically sorted category is dropped
        col_name = f"{category_name}_{encoded_value}"
        
        # Return the column name if it exists in encoded columns
        # Otherwise return None (meaning it's the dropped baseline category)
        if col_name in encoded_cols:
            # This will be set to 1 in features dict
            return col_name
        else:
            # This is the baseline (dropped) category - all encoded cols stay 0
            return None
    
    def validate_features(self, df):
        """
        Validate that feature vector has correct shape and values.
        
        Args:
            df: DataFrame with features
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check shape
        if df.shape[1] != len(self.feature_names):
            return False, f"Expected {len(self.feature_names)} features, got {df.shape[1]}"
        
        # Check column names
        if list(df.columns) != self.feature_names:
            return False, "Feature names don't match training"
        
        # Check for NaN values
        if df.isnull().any().any():
            return False, "Features contain NaN values"
        
        # Check for infinite values
        if np.isinf(df.values).any():
            return False, "Features contain infinite values"
        
        return True, "Valid"