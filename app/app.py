from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import load_model

app = FastAPI(title="Fraud Detection API")

try:
    model, scaler = load_model("models/model.joblib", "models/scaler.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {str(e)}")

class Transaction(BaseModel):
    data: list

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. Use POST /predict to submit data."}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        input_data = np.array(transaction.data).reshape(1, -1)
        if input_data.shape[1] != scaler.mean_.shape[0]:
            raise ValueError(f"Expected {scaler.mean_.shape[0]} features, got {input_data.shape[1]}")
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        is_fraud = bool(prediction[0] == -1)
        return {"fraud": is_fraud}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
