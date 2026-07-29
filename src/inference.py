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
    Takes raw patient data, engineers features, and returns Sepsis predictions.
    """
    df = patient_data.copy()
    
    # 1. Ensure ALL expected columns exist and fill with medians
    for col, median_val in global_medians.items():
        if col not in df.columns:
            # If the API request missed this column entirely, create it using the median
            df[col] = median_val
        else:
            # If it exists, forward-fill historical data, then fill remaining blanks with median
            df[col] = df.groupby('Patient_ID')[col].ffill()
            df[col] = df[col].fillna(median_val)
        
    # 2. Apply Feature Engineering
    df['System_Overload_Score'] = ((df['HR'] > 100).astype(int) + (df['SBP'] < 90).astype(int) + (df['Resp'] > 22).astype(int) + ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(int))

    df['Age_Frailty_Index'] = (df['Age'] / 100) * df['System_Overload_Score']

    df['FiO2_4hr_mean'] = df.groupby('Patient_ID')['FiO2'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['FiO2_4hr_std'] = df.groupby('Patient_ID')['FiO2'].transform(lambda x: x.rolling(window=4, min_periods=2).std())

    df['Temp_4hr_mean'] = df.groupby('Patient_ID')['Temp'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

    o2sat_baseline = df.groupby('Patient_ID')['O2Sat'].transform('first')
    df['O2Sat_Admission_Delta'] = df['O2Sat'] - o2sat_baseline

    df['SBP_4hr_mean'] = df.groupby('Patient_ID')['SBP'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['SBP_4hr_std'] = df.groupby('Patient_ID')['SBP'].transform(lambda x: x.rolling(window=4, min_periods=2).std())
    df['SBP_Volatility'] = df['SBP_4hr_std'] / (df['SBP_4hr_mean'] + 1e-5)

    df['ICU_Hour'] = range(1, len(df) + 1)
    
    # 3. Predict
    X = df[model.feature_names_in_]
    probabilities = model.predict_proba(X)[:, 1]
    
    # 4. Return results as a dictionary
    return {
        "patient_id": str(df['Patient_ID'].iloc[0]),
        "sepsis_risk_probability": float(probabilities[-1]), # Return the risk for the most recent hour
        "alarm_triggered": bool(probabilities[-1] >= 0.6260)
    }