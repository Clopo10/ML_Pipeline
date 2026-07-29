from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.inference import run_inference

app = FastAPI(title="Sepsis Prediction API", version="1.0")

# Define the expected incoming JSON format
class PatientPayload(BaseModel):
    data: list[dict] # Expects a list of dictionaries (hourly patient logs)

@app.post("/predict")
def predict_sepsis(payload: PatientPayload):
    try:
        # Convert the incoming JSON payload into a Pandas DataFrame
        df = pd.DataFrame(payload.data)
        
        # Run your ML pipeline
        result = run_inference(df)
        
        return {"status": "success", "prediction": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))