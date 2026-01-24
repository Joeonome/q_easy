"""
Test script to validate the fixed implementation
Run this after training to ensure everything works correctly
"""

import joblib
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer

def test_files_exist():
    """Test 1: Check all required files exist"""
    print("="*80)
    print("TEST 1: Checking Required Files")
    print("="*80)
    
    required_files = [
        'best_lgbm_model.pkl',
        'scaler.pkl',
        'feature_names.pkl',
        'model_metadata.pkl'
    ]
    
    all_exist = True
    for file in required_files:
        try:
            with open(file, 'rb'):
                print(f"✅ {file} - Found")
        except FileNotFoundError:
            print(f"❌ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_load_files():
    """Test 2: Check files can be loaded"""
    print("\n" + "="*80)
    print("TEST 2: Loading Files")
    print("="*80)
    
    try:
        model = joblib.load('best_lgbm_model.pkl')
        print("✅ Model loaded successfully")
        
        scaler = joblib.load('scaler.pkl')
        print("✅ Scaler loaded successfully")
        
        feature_names = joblib.load('feature_names.pkl')
        print(f"✅ Feature names loaded ({len(feature_names)} features)")
        
        metadata = joblib.load('model_metadata.pkl')
        print(f"✅ Metadata loaded")
        print(f"   - Model R²: {metadata.get('model_r2', 'N/A'):.4f}")
        print(f"   - Model RMSE: {metadata.get('model_rmse', 'N/A'):.4f}")
        print(f"   - Numerical cols scaled: {len(metadata.get('numerical_cols_scaled', []))}")
        print(f"   - Categorical cols: {len(metadata.get('categorical_cols', []))}")
        
        return model, scaler, feature_names, metadata
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None, None

def test_feature_engineer(metadata, scaler):
    """Test 3: Test FeatureEngineer class"""
    print("\n" + "="*80)
    print("TEST 3: Testing FeatureEngineer")
    print("="*80)
    
    try:
        fe = FeatureEngineer(metadata, scaler)
        print("✅ FeatureEngineer initialized successfully")
        
        # Test with sample inputs
        test_inputs = {
            'age': 45,
            'gender': 'Male',
            'temp': 37.0,
            'hr': 75,
            'rr': 16,
            'bp_sys': 120,
            'bp_dia': 80,
            'spo2': 98,
            'pain': 3,
            'service': 'Emergency Medicine',
            'complaint': 'Chest Pain',
            'danger_signs': ['None'],
            'triage_level': 3,
            'is_pregnant': False,
            'occupancy': 75,
            'doctor_load': 8,
            'shift_doctors': 5,
            'shift_nurses': 10,
            'shift_triage': 2,
            'service_patients': 15,
            'service_queue': 10,
            'service_occupancy': 70,
            'mri_avail': True,
            'xray_avail': True,
            'or_avail': True,
            'outside_temp': 28.0,
            'arrival_channel': 'Walk-in',
            'shift': 'Morning',
            'season': 'Summer',
            'weather': 'Clear'
        }
        
        X = fe.create_features(test_inputs)
        print(f"✅ Features created: shape {X.shape}")
        
        is_valid, msg = fe.validate_features(X)
        if is_valid:
            print(f"✅ Feature validation passed: {msg}")
        else:
            print(f"❌ Feature validation failed: {msg}")
            return None
        
        return fe, X
        
    except Exception as e:
        print(f"❌ Error in FeatureEngineer: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_prediction(model, fe):
    """Test 4: Test prediction"""
    print("\n" + "="*80)
    print("TEST 4: Testing Prediction")
    print("="*80)
    
    try:
        # Multiple test cases
        test_cases = [
            {
                'name': 'Low Priority Patient',
                'inputs': {
                    'age': 30,
                    'gender': 'Female',
                    'temp': 36.8,
                    'hr': 70,
                    'rr': 14,
                    'bp_sys': 115,
                    'bp_dia': 75,
                    'spo2': 99,
                    'pain': 2,
                    'service': 'Emergency Medicine',
                    'complaint': 'Headache',
                    'danger_signs': ['None'],
                    'triage_level': 4,
                    'is_pregnant': False,
                    'occupancy': 50,
                    'doctor_load': 5,
                    'shift_doctors': 6,
                    'shift_nurses': 12,
                    'shift_triage': 3,
                    'service_patients': 10,
                    'service_queue': 5,
                    'service_occupancy': 40,
                    'mri_avail': True,
                    'xray_avail': True,
                    'or_avail': True,
                    'outside_temp': 25.0,
                    'arrival_channel': 'Walk-in',
                    'shift': 'Morning',
                    'season': 'Spring',
                    'weather': 'Clear'
                }
            },
            {
                'name': 'High Priority Patient',
                'inputs': {
                    'age': 75,
                    'gender': 'Male',
                    'temp': 39.5,
                    'hr': 110,
                    'rr': 24,
                    'bp_sys': 165,
                    'bp_dia': 95,
                    'spo2': 92,
                    'pain': 8,
                    'service': 'Cardiology',
                    'complaint': 'Chest Pain',
                    'danger_signs': ['Difficulty Breathing'],
                    'triage_level': 1,
                    'is_pregnant': False,
                    'occupancy': 90,
                    'doctor_load': 12,
                    'shift_doctors': 4,
                    'shift_nurses': 8,
                    'shift_triage': 2,
                    'service_patients': 25,
                    'service_queue': 20,
                    'service_occupancy': 95,
                    'mri_avail': True,
                    'xray_avail': True,
                    'or_avail': False,
                    'outside_temp': 30.0,
                    'arrival_channel': 'Ambulance',
                    'shift': 'Night',
                    'season': 'Summer',
                    'weather': 'Clear'
                }
            }
        ]
        
        for test_case in test_cases:
            print(f"\nTesting: {test_case['name']}")
            X = fe.create_features(test_case['inputs'])
            
            # Predict
            log_wait = model.predict(X)[0]
            wait_minutes = np.exp(log_wait)
            
            print(f"  Predicted wait time: {wait_minutes:.0f} minutes")
            
            # Check if prediction is reasonable
            if 5 <= wait_minutes <= 500:
                print(f"  ✅ Prediction is in reasonable range")
            else:
                print(f"  ⚠️  Warning: Prediction seems unusual ({wait_minutes:.0f} min)")
        
        print("\n✅ Predictions completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_consistency(metadata, fe):
    """Test 5: Verify feature consistency"""
    print("\n" + "="*80)
    print("TEST 5: Feature Consistency Check")
    print("="*80)
    
    try:
        # Create features
        test_inputs = {
            'age': 45,
            'gender': 'Male',
            'temp': 37.0,
            'hr': 75,
            'rr': 16,
            'bp_sys': 120,
            'bp_dia': 80,
            'spo2': 98,
            'pain': 3,
            'service': 'Emergency Medicine',
            'complaint': 'Fever',
            'danger_signs': ['None'],
            'triage_level': 3,
            'is_pregnant': False,
            'occupancy': 75,
            'doctor_load': 8,
            'shift_doctors': 5,
            'shift_nurses': 10,
            'shift_triage': 2,
            'service_patients': 15,
            'service_queue': 10,
            'service_occupancy': 70,
            'mri_avail': True,
            'xray_avail': True,
            'or_avail': True,
            'outside_temp': 28.0,
            'arrival_channel': 'Walk-in',
            'shift': 'Morning',
            'season': 'Summer',
            'weather': 'Clear'
        }
        
        X = fe.create_features(test_inputs)
        
        # Check feature count
        expected_count = len(metadata['feature_names'])
        actual_count = X.shape[1]
        
        if expected_count == actual_count:
            print(f"✅ Feature count matches: {actual_count}")
        else:
            print(f"❌ Feature count mismatch: expected {expected_count}, got {actual_count}")
            return False
        
        # Check column names
        if list(X.columns) == metadata['feature_names']:
            print(f"✅ Feature names match training")
        else:
            print(f"❌ Feature names don't match training")
            print(f"   Missing: {set(metadata['feature_names']) - set(X.columns)}")
            print(f"   Extra: {set(X.columns) - set(metadata['feature_names'])}")
            return False
        
        # Check for NaN
        if not X.isnull().any().any():
            print(f"✅ No NaN values in features")
        else:
            print(f"❌ Found NaN values in features")
            print(X.isnull().sum()[X.isnull().sum() > 0])
            return False
        
        # Check for infinite values
        if not np.isinf(X.values).any():
            print(f"✅ No infinite values in features")
        else:
            print(f"❌ Found infinite values in features")
            return False
        
        # Check data types
        print(f"✅ All consistency checks passed")
        return True
        
    except Exception as e:
        print(f"❌ Error in consistency check: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n")
    print("🏥 Q-EASY IMPLEMENTATION VALIDATION")
    print("="*80)
    
    # Test 1: Files exist
    if not test_files_exist():
        print("\n❌ FAILED: Required files missing. Please run training script first.")
        return
    
    # Test 2: Load files
    model, scaler, feature_names, metadata = test_load_files()
    if model is None:
        print("\n❌ FAILED: Could not load files")
        return
    
    # Test 3: FeatureEngineer
    fe, X = test_feature_engineer(metadata, scaler)
    if fe is None:
        print("\n❌ FAILED: FeatureEngineer initialization failed")
        return
    
    # Test 4: Predictions
    if not test_prediction(model, fe):
        print("\n❌ FAILED: Prediction test failed")
        return
    
    # Test 5: Consistency
    if not test_feature_consistency(metadata, fe):
        print("\n❌ FAILED: Feature consistency check failed")
        return
    
    # Summary
    print("\n" + "="*80)
    print("🎉 ALL TESTS PASSED!")
    print("="*80)
    print("\nYour implementation is ready to use!")
    print("\nNext steps:")
    print("  1. Run: streamlit run app_fixed.py")
    print("  2. Test the app with various inputs")
    print("  3. Monitor predictions for accuracy")
    print("\n")

if __name__ == "__main__":
    main()