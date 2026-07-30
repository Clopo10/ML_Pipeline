from fastapi import FastAPI, HTTPException
import pandas as pd
from src.inference import run_inference

app = FastAPI(title="Sepsis Prediction API", version="1.0")

@app.post("/predict")
def predict_sepsis(payload: list[dict]): 
    try:
        # Convert the incoming JSON array directly into a Pandas DataFrame
        df = pd.DataFrame(payload)
        
        # Run your ML pipeline
        result = run_inference(df)
        
        return {"status": "success", "prediction": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))