
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (CustomerFeatures, HealthResponse, ModelInfoResponse,
                          PredictionResponse)

MODEL_PATH = Path("models/churn_model.joblib")
MODEL_INFO_PATH = Path("models/model_info.json")

app = FastAPI(
    title="E-commerce Customer Churn Prediction API",
    description="Predicts whether a customer is likely to churn based on RFM behavior features.",
    version="1.0.0",
)

model = None
model_info = None


@app.on_event("startup")
def load_model():
    global model, model_info
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python src/models/train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    with open(MODEL_INFO_PATH) as f:
        model_info = json.load(f)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok" if model is not None else "model not loaded"}


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model info not loaded")
    return {
        "model_type": model_info["model_type"],
        "best_params": model_info["best_params"],
        "metrics": model_info["metrics"],
        "features": model_info["features"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    return {"churn_prediction": prediction, "churn_probability": round(probability, 4)}
