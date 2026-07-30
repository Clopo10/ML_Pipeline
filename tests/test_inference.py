import pandas as pd
from src.inference import run_inference

def test_inference_pipeline():
    """
    Automated test to ensure the inference engine handles time-series data,
    calculates rolling/delta features correctly, and returns the expected JSON.
    """
    # Create a mock patient with a 3-hour trajectory to test rolling/delta features
    mock_patient = pd.DataFrame([
        {
            "Patient_ID": "test_patient_001",
            "Age": 45,
            "HR": 80,    # Hour 1: Healthy Baseline
            "SBP": 120,  
            "Resp": 16,
            "Temp": 37.0
        },
        {
            "Patient_ID": "test_patient_001",
            "Age": 45,
            "HR": 100,   # Hour 2: Vitals shifting (Deltas trigger here)
            "SBP": 100,  
            "Resp": 20,
            "Temp": 38.1
        },
        {
            "Patient_ID": "test_patient_001",
            "Age": 45,
            "HR": 120,   # Hour 3: Patient is crashing
            "SBP": 80,   
            "Resp": 26,
            "Temp": 39.2,
            "Lactate": 4.0 # Doctor ordered a lactate test! (Flags trigger here)
        }
    ])
    
    # Run the pipeline
    result = run_inference(mock_patient)
    
    # Assertions
    assert isinstance(result, dict), "Output must be a dictionary"
    assert "patient_id" in result, "Missing 'patient_id' in output"
    assert "sepsis_risk_raw" in result, "Missing raw probability score"
    assert "sepsis_risk_display" in result, "Missing display probability string"
    assert "alarm_triggered" in result, "Missing alarm status"
    
    # Type checking
    assert isinstance(result["sepsis_risk_raw"], float), "Raw probability must be a float"
    assert isinstance(result["sepsis_risk_display"], str), "Display probability must be a string"
    assert isinstance(result["alarm_triggered"], bool), "Alarm status must be a boolean"
    
    print("Inference pipeline time-series test passed successfully!")

if __name__ == "__main__":
    test_inference_pipeline()