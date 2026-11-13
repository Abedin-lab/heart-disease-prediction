"""Automated tests for the Heart Disease Prediction FastAPI endpoints."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "RandomForestClassifier"
    assert len(data["features"]) == 13
    assert "age" in data["features"]


def test_predict_positive():
    payload = {
        "age": 58,
        "sex": 0,
        "cp": 0,
        "trestbps": 100,
        "chol": 248,
        "fbs": 0,
        "restecg": 0,
        "thalach": 122,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"heart_disease": True}


def test_predict_negative():
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"heart_disease": False}


if __name__ == "__main__":
    test_health()
    test_info()
    test_predict_positive()
    test_predict_negative()
    print("All API endpoint tests (/health, /info, /predict) passed successfully!")
