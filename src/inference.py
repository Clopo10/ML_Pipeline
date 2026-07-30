import pandas as pd
import xgboost as xgb
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tuned_model.json')
MEDIANS_PATH = os.path.join(BASE_DIR, 'data', 'cleaned', 'hospital_A_medians.json')

# Pre-load the model and medians into memory when the app starts
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

with open(MEDIANS_PATH, 'r') as f:
    global_medians = json.load(f)

def run_inference(patient_data: pd.DataFrame) -> dict:
    """
    Takes raw patient data (history up to current hour), engineers features, and returns Sepsis predictions.
    """
    df = patient_data.copy()
    
    # ==========================================
    # 1. MISSINGNESS FLAGS
    # ==========================================
    # If the API payload doesn't include these columns at all, the count is 0.
    if 'Lactate' in df.columns:
        df['Lactate_Order_Count'] = df.groupby('Patient_ID')['Lactate'].transform(lambda x: x.notnull().cumsum())
    else:
        df['Lactate_Order_Count'] = 0

    if 'WBC' in df.columns:
        df['WBC_Order_Count'] = df.groupby('Patient_ID')['WBC'].transform(lambda x: x.notnull().cumsum())
    else:
        df['WBC_Order_Count'] = 0

    # ==========================================
    # 2. SAFE IMPUTATION
    # ==========================================
    for col, median_val in global_medians.items():
        if col not in df.columns:
            # If the API request missed this column entirely, create it using the median
            df[col] = median_val
        else:
            # If it exists, forward-fill historical data, then fill remaining blanks with median
            df[col] = df.groupby('Patient_ID')[col].ffill()
            df[col] = df[col].fillna(median_val)
        
    # ==========================================
    # 3. FEATURE ENGINEERING
    # ==========================================
    # Composite Scores
    df['System_Overload_Score'] = ((df['HR'] > 100).astype(int) + (df['SBP'] < 90).astype(int) + (df['Resp'] > 22).astype(int) + ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(int))
    df['Age_Frailty_Index'] = (df['Age'] / 100) * df['System_Overload_Score']

    # Vital Sign Dynamics (FiO2, Temp, SBP)
    df['FiO2_4hr_mean'] = df.groupby('Patient_ID')['FiO2'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['FiO2_4hr_std'] = df.groupby('Patient_ID')['FiO2'].transform(lambda x: x.rolling(window=4, min_periods=2).std())
    df['Temp_4hr_mean'] = df.groupby('Patient_ID')['Temp'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['SBP_4hr_mean'] = df.groupby('Patient_ID')['SBP'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['SBP_4hr_std'] = df.groupby('Patient_ID')['SBP'].transform(lambda x: x.rolling(window=4, min_periods=2).std())
    df['SBP_Volatility'] = df['SBP_4hr_std'] / (df['SBP_4hr_mean'] + 1e-5)

    # Domain-Agnostic Deltas (The Domain Shift Cure)
    o2sat_baseline = df.groupby('Patient_ID')['O2Sat'].transform('first')
    hr_baseline = df.groupby('Patient_ID')['HR'].transform('first')
    resp_baseline = df.groupby('Patient_ID')['Resp'].transform('first')
    map_baseline = df.groupby('Patient_ID')['MAP'].transform('first')

    df['O2Sat_Admission_Delta'] = df['O2Sat'] - o2sat_baseline
    df['HR_Admission_Delta'] = df['HR'] - hr_baseline
    df['Resp_Admission_Delta'] = df['Resp'] - resp_baseline
    df['MAP_Admission_Delta'] = df['MAP'] - map_baseline

    # New High-Frequency Rolling Vitals
    df['HR_4hr_mean'] = df.groupby('Patient_ID')['HR'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['HR_4hr_std'] = df.groupby('Patient_ID')['HR'].transform(lambda x: x.rolling(window=4, min_periods=2).std())
    df['MAP_4hr_mean'] = df.groupby('Patient_ID')['MAP'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

    # Organ Failure Ratio
    bili_baseline = df.groupby('Patient_ID')['Bilirubin_total'].transform('first')
    df['Bilirubin_Admission_Delta'] = df['Bilirubin_total'] - bili_baseline

    # Safely calculate ICU hour based on the patient's incoming timeline
    df['ICU_Hour'] = df.groupby('Patient_ID').cumcount() + 1
    
    # ==========================================
    # 4. PREDICT & RETURN
    # ==========================================
    X = df[model.feature_names_in_]
    probabilities = model.predict_proba(X)[:, 1]
    prob = float(probabilities[-1])
    
    LOCKED_THRESHOLD = 0.2475

    return {
        "patient_id": str(df['Patient_ID'].iloc[0]),
        "sepsis_risk_raw": prob,                                # The pure math (for software)
        "sepsis_risk_display": f"{prob:.4f} ({prob * 100:.1f}%)", # The formatted text (for humans)
        "alarm_triggered": bool(prob >= LOCKED_THRESHOLD)
    }