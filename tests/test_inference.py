import pandas as pd
from src.inference import run_inference

def test_inference_pipeline():
    """
    Automated test to ensure the inference engine handles raw data 
    and returns the correct JSON keys without crashing.
    """
    # Create a minimal mock patient dataframe
    mock_patient = pd.DataFrame([
        {
            "Patient_ID": "test_patient_001",
            "Age": 45,
            "HR": 120,
            "SBP": 80,
            "Resp": 26,
            "Temp": 39.2
        }
    ])
    
    # Run the pipeline
    result = run_inference(mock_patient)
    
    # Assertions (Prove the output is structurally sound)
    assert isinstance(result, dict), "Output must be a dictionary"
    assert "patient_id" in result, "Missing 'patient_id' in output"
    assert "sepsis_risk_probability" in result, "Missing probability score"
    assert "alarm_triggered" in result, "Missing alarm status"
    assert isinstance(result["sepsis_risk_probability"], float), "Probability must be a float"
    assert isinstance(result["alarm_triggered"], bool), "Alarm status must be a boolean"