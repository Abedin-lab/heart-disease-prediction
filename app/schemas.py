"""Pydantic schemas for Heart Disease Prediction API."""

from typing import List
from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    """Input features for heart disease prediction."""

    age: int = Field(..., ge=1, le=120, description="Age in years", examples=[58])
    sex: int = Field(
        ..., ge=0, le=1, description="Sex (1 = male; 0 = female)", examples=[0]
    )
    cp: int = Field(
        ...,
        ge=0,
        le=3,
        description="Chest pain type (0-3)",
        examples=[0],
    )
    trestbps: int = Field(
        ...,
        ge=50,
        le=300,
        description="Resting blood pressure (in mm Hg on admission to the hospital)",
        examples=[100],
    )
    chol: int = Field(
        ...,
        ge=80,
        le=700,
        description="Serum cholesterol in mg/dl",
        examples=[248],
    )
    fbs: int = Field(
        ...,
        ge=0,
        le=1,
        description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
        examples=[0],
    )
    restecg: int = Field(
        ...,
        ge=0,
        le=2,
        description="Resting electrocardiographic results (0-2)",
        examples=[0],
    )
    thalach: int = Field(
        ...,
        ge=50,
        le=250,
        description="Maximum heart rate achieved",
        examples=[122],
    )
    exang: int = Field(
        ...,
        ge=0,
        le=1,
        description="Exercise induced angina (1 = yes; 0 = no)",
        examples=[0],
    )
    oldpeak: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="ST depression induced by exercise relative to rest",
        examples=[1.0],
    )
    slope: int = Field(
        ...,
        ge=0,
        le=2,
        description="The slope of the peak exercise ST segment (0-2)",
        examples=[1],
    )
    ca: int = Field(
        ...,
        ge=0,
        le=4,
        description="Number of major vessels (0-4) colored by flourosopy",
        examples=[0],
    )
    thal: int = Field(
        ...,
        ge=0,
        le=3,
        description="Thalassemia (0-3: 1 = normal; 2 = fixed defect; 3 = reversable defect)",
        examples=[2],
    )


class PredictionOutput(BaseModel):
    """Prediction output indicating presence or absence of heart disease."""

    heart_disease: bool = Field(
        ..., description="True if heart disease is predicted to be present, False otherwise"
    )


class HealthOutput(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Service health status")


class ModelInfoOutput(BaseModel):
    """Model metadata and input feature information."""

    model_type: str = Field(..., description="Type of trained machine learning model")
    features: List[str] = Field(..., description="Ordered list of input features")
    target_classes: dict = Field(..., description="Mapping of target values")
