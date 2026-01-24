import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import joblib

# Load the Excel file
try:
    df_wait_time = pd.read_excel('./master_df_hospital_waittime.xlsx')
    print("Successfully loaded master_df_hospital_waittime.xlsx")
    print(f"\nDataFrame shape: {df_wait_time.shape}")
except FileNotFoundError:
    print("Error: The file 'master_df_hospital_waittime.xlsx' was not found.")
    exit()

print("\n" + "="*80)
print("STEP 1: HANDLE MISSING VALUES")
print("="*80)

print("\nNumber of missing values before imputation:")
missing_cols = ['vital_temp_c', 'vital_hr_bpm', 'vital_rr_bpm', 
                'vital_bp_systolic', 'vital_bp_diastolic', 'vital_spo2', 'pain_score_0_10']
print(df_wait_time[missing_cols].isnull().sum())

# Impute missing values with median
for col in missing_cols:
    if df_wait_time[col].isnull().any():
        median_val = df_wait_time[col].median()
        df_wait_time[col].fillna(median_val, inplace=True)
        print(f"Imputed '{col}' with median: {median_val:.2f}")

print("\n" + "="*80)
print("STEP 2: ONE-HOT ENCODING")
print("="*80)

# Store original categorical values for metadata
categorical_cols = [
    'service',
    'primary_complaint',
    'danger_signs',
    'risk_flags',
    'arrival_channel',
    'gender',
    'shift_name',
    'seasonality_factor',
    'weather_condition'
]

# Store original unique values before encoding
original_categories = {}
for col in categorical_cols:
    original_categories[col] = sorted(df_wait_time[col].unique().tolist())
    print(f"\n{col}: {original_categories[col]}")

print(f"\nDataFrame shape before one-hot encoding: {df_wait_time.shape}")

# Apply one-hot encoding
df_encoded = pd.get_dummies(df_wait_time, columns=categorical_cols, drop_first=True, dtype=int)

# Get the columns that were created by one-hot encoding
encoded_columns = {}
for col in categorical_cols:
    encoded_columns[col] = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
    print(f"\n{col} encoded to: {encoded_columns[col]}")

print(f"\nDataFrame shape after one-hot encoding: {df_encoded.shape}")

print("\n" + "="*80)
print("STEP 3: TIME-BASED FEATURE ENGINEERING")
print("="*80)

# Extract month
if 'month' not in df_encoded.columns:
    df_encoded['month'] = df_encoded['arrival_datetime'].dt.month
    print("Added 'month' feature")

# Convert datetime to timestamp
df_encoded['arrival_timestamp'] = df_encoded['arrival_datetime'].astype('int64') // 10**9
df_encoded['first_doctor_timestamp'] = df_encoded['first_doctor_datetime'].astype('int64') // 10**9

# Drop original datetime columns
df_encoded.drop(columns=['arrival_datetime', 'first_doctor_datetime'], inplace=True)

print(f"DataFrame shape after time features: {df_encoded.shape}")

print("\n" + "="*80)
print("STEP 4: PREPARE FEATURES AND TARGET")
print("="*80)

# Define features and target
X = df_encoded.drop(columns=['visit_id', 'wait_minutes', 'log_wait_minutes', 
                              'first_doctor_timestamp', 'total_time_in_system_minutes'])
y = df_encoded['log_wait_minutes']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify numerical columns for scaling
numerical_cols_to_scale = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude binary columns (those with only 0 and 1)
binary_cols = [col for col in numerical_cols_to_scale if X[col].nunique() <= 2]
numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col not in binary_cols]

print(f"\nNumerical columns to scale ({len(numerical_cols_to_scale)}): {numerical_cols_to_scale}")
print(f"Binary columns (not scaled, {len(binary_cols)}): {binary_cols[:10]}...")

print("\n" + "="*80)
print("STEP 5: SCALING")
print("="*80)

# Initialize and fit scaler
scaler = StandardScaler()
X[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])

print("Scaling completed")

print("\n" + "="*80)
print("STEP 6: TRAIN-TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

print("\n" + "="*80)
print("STEP 7: TRAIN LINEAR REGRESSION (BASELINE)")
print("="*80)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression Results:")
print(f"  MAE: {mae_linear:.4f}")
print(f"  RMSE: {rmse_linear:.4f}")
print(f"  R²: {r2_linear:.4f} ({r2_linear:.2%})")

print("\n" + "="*80)
print("STEP 8: TRAIN LIGHTGBM WITH HYPERPARAMETER TUNING")
print("="*80)

lgbm_initial = lgb.LGBMRegressor(
    n_estimators=150,
    max_depth=6,
    min_child_samples=20,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    force_row_wise=True,
    random_state=42,
    verbose=-1
)

param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.07],
    "min_child_samples": [15, 20, 30],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8]
}

random_search = RandomizedSearchCV(
    estimator=lgbm_initial,
    param_distributions=param_dist,
    n_iter=12,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Running hyperparameter search...")
random_search.fit(X_train, y_train)

print("\nBest parameters:", random_search.best_params_)
print(f"Best CV RMSE: {np.sqrt(abs(random_search.best_score_)):.4f}")

best_lgbm_model = random_search.best_estimator_

print("\n" + "="*80)
print("STEP 9: EVALUATE BEST MODEL")
print("="*80)

y_pred_lgbm = best_lgbm_model.predict(X_test)

mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print("LightGBM Results:")
print(f"  MAE: {mae_lgbm:.4f}")
print(f"  RMSE: {rmse_lgbm:.4f}")
print(f"  R²: {r2_lgbm:.4f} ({r2_lgbm:.2%})")

print("\n" + "="*80)
print("STEP 10: SAVE MODEL AND METADATA")
print("="*80)

# Save the model
joblib.dump(best_lgbm_model, 'best_lgbm_model.pkl')
print("✓ Saved: best_lgbm_model.pkl")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("✓ Saved: scaler.pkl")

# Save feature names (in exact order)
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print(f"✓ Saved: feature_names.pkl ({len(feature_names)} features)")

# Save comprehensive metadata
metadata = {
    'feature_names': feature_names,
    'numerical_cols_scaled': numerical_cols_to_scale,
    'binary_cols': binary_cols,
    'categorical_cols': categorical_cols,
    'original_categories': original_categories,
    'encoded_columns': encoded_columns,
    'model_rmse': rmse_lgbm,
    'model_r2': r2_lgbm,
    'target_variable': 'log_wait_minutes',
    'imputation_values': {col: df_wait_time[col].median() for col in missing_cols}
}

joblib.dump(metadata, 'model_metadata.pkl')
print("✓ Saved: model_metadata.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nFinal Model Performance:")
print(f"  R² Score: {r2_lgbm:.4f} ({r2_lgbm:.2%})")
print(f"  RMSE: {rmse_lgbm:.4f}")
print(f"  MAE: {mae_lgbm:.4f}")
print(f"\nNumber of features: {len(feature_names)}")
print("\nFiles saved:")
print("  - best_lgbm_model.pkl")
print("  - scaler.pkl")
print("  - feature_names.pkl")
print("  - model_metadata.pkl")