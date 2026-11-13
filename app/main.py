"""FastAPI application serving Heart Disease predictions."""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    HealthOutput,
    HeartDiseaseInput,
    ModelInfoOutput,
    PredictionOutput,
)

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    description="A FastAPI service that predicts the presence of heart disease using a trained machine learning classifier.",
    version="1.0.0",
)

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


@app.get("/", tags=["Root"])
def read_root():
    """Root welcome endpoint."""
    return {
        "message": "Welcome to the Heart Disease Prediction API. Visit /docs for Swagger UI."
    }


@app.get("/health", response_model=HealthOutput, tags=["Monitoring"])
def health_check():
    """Check if the API and model are healthy and loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthOutput(status="healthy")


@app.get("/info", response_model=ModelInfoOutput, tags=["Metadata"])
def model_info():
    """Return metadata about the deployed model and expected input features."""
    model_type = type(model).__name__ if model is not None else "Unloaded"
    return ModelInfoOutput(
        model_type=model_type,
        features=FEATURE_NAMES,
        target_classes={"0": "absence (false)", "1": "presence (true)"},
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_heart_disease(payload: HeartDiseaseInput):
    """Predict whether the patient has heart disease (true/false)."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Please run train_model.py first.",
        )

    input_df = pd.DataFrame([payload.model_dump()])[FEATURE_NAMES]
    prediction = model.predict(input_df)[0]
    return PredictionOutput(heart_disease=bool(int(prediction) == 1))
