"""FastAPI application serving Heart Disease predictions."""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse, tags=["Root"])
def read_root():
    """Root interactive web UI for Heart Disease Prediction."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Heart Disease Prediction API</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --danger: #ef4444;
      --success: #22c55e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .container { max-width: 860px; margin: 0 auto; }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1.75rem;
      padding-bottom: 1.25rem;
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 1.6rem; font-weight: 700; }
    .subtitle { color: var(--muted); font-size: 0.92rem; margin-top: 0.25rem; }
    .nav-links { display: flex; gap: 0.6rem; flex-wrap: wrap; }
    .btn-link {
      text-decoration: none;
      padding: 0.5rem 0.9rem;
      border-radius: 0.5rem;
      font-size: 0.85rem;
      font-weight: 600;
      border: 1px solid var(--border);
      color: var(--text);
      background: var(--card);
      transition: all 0.15s;
    }
    .btn-link.primary {
      background: var(--accent);
      color: #0f172a;
      border-color: var(--accent);
    }
    .btn-link:hover { opacity: 0.9; transform: translateY(-1px); }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.85rem;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .presets {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }
    .preset-btns { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    button {
      cursor: pointer;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      padding: 0.45rem 0.85rem;
      border-radius: 0.45rem;
      font-size: 0.82rem;
      font-weight: 500;
    }
    button:hover { border-color: var(--accent); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 1rem;
    }
    .field label {
      display: block;
      font-size: 0.8rem;
      color: var(--muted);
      margin-bottom: 0.35rem;
      font-weight: 500;
    }
    .field input, .field select {
      width: 100%;
      padding: 0.55rem 0.7rem;
      border-radius: 0.45rem;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font-size: 0.92rem;
    }
    .submit-btn {
      margin-top: 1.5rem;
      width: 100%;
      padding: 0.8rem;
      background: var(--accent);
      color: #0f172a;
      border: none;
      border-radius: 0.5rem;
      font-size: 1rem;
      font-weight: 700;
    }
    #result {
      display: none;
      padding: 1.25rem;
      border-radius: 0.75rem;
      text-align: center;
      font-size: 1.15rem;
      font-weight: 600;
    }
    .res-true {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid var(--danger);
      color: #fca5a5;
    }
    .res-false {
      background: rgba(34, 197, 94, 0.15);
      border: 1px solid var(--success);
      color: #86efac;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Heart Disease Prediction</h1>
        <p class="subtitle">RandomForestClassifier API &bull; Interactive Clinical Predictor</p>
      </div>
      <div class="nav-links">
        <a href="/docs" class="btn-link primary">Swagger UI (/docs)</a>
        <a href="/redoc" class="btn-link">ReDoc (/redoc)</a>
        <a href="/health" class="btn-link">Health</a>
        <a href="/info" class="btn-link">Model Info</a>
      </div>
    </header>

    <div class="card">
      <div class="presets">
        <span style="font-weight: 600; font-size: 0.95rem;">Patient Clinical Parameters</span>
        <div class="preset-btns">
          <button type="button" onclick="loadSample(true)">Load Positive Sample</button>
          <button type="button" onclick="loadSample(false)">Load Negative Sample</button>
        </div>
      </div>

      <form id="predictForm">
        <div class="grid">
          <div class="field"><label>Age (years)</label><input type="number" id="age" value="58" required /></div>
          <div class="field">
            <label>Sex</label>
            <select id="sex"><option value="1">1 - Male</option><option value="0" selected>0 - Female</option></select>
          </div>
          <div class="field">
            <label>Chest Pain Type (cp: 0–3)</label>
            <input type="number" id="cp" min="0" max="3" value="0" required />
          </div>
          <div class="field"><label>Resting BP (trestbps mm Hg)</label><input type="number" id="trestbps" value="100" required /></div>
          <div class="field"><label>Cholesterol (chol mg/dl)</label><input type="number" id="chol" value="248" required /></div>
          <div class="field">
            <label>Fasting Blood Sugar &gt; 120 (fbs)</label>
            <select id="fbs"><option value="0" selected>0 - False</option><option value="1">1 - True</option></select>
          </div>
          <div class="field"><label>Resting ECG (restecg: 0–2)</label><input type="number" id="restecg" min="0" max="2" value="0" required /></div>
          <div class="field"><label>Max Heart Rate (thalach)</label><input type="number" id="thalach" value="122" required /></div>
          <div class="field">
            <label>Exercise Induced Angina (exang)</label>
            <select id="exang"><option value="0" selected>0 - No</option><option value="1">1 - Yes</option></select>
          </div>
          <div class="field"><label>ST Depression (oldpeak)</label><input type="number" step="0.1" id="oldpeak" value="1.0" required /></div>
          <div class="field"><label>ST Slope (slope: 0–2)</label><input type="number" id="slope" min="0" max="2" value="1" required /></div>
          <div class="field"><label>Major Vessels (ca: 0–4)</label><input type="number" id="ca" min="0" max="4" value="0" required /></div>
          <div class="field"><label>Thalassemia (thal: 0–3)</label><input type="number" id="thal" min="0" max="3" value="2" required /></div>
        </div>
        <button type="submit" class="submit-btn">Predict Heart Disease Risk</button>
      </form>
    </div>

    <div id="result"></div>
  </div>

  <script>
    const samples = {
      positive: { age: 58, sex: 0, cp: 0, trestbps: 100, chol: 248, fbs: 0, restecg: 0, thalach: 122, exang: 0, oldpeak: 1.0, slope: 1, ca: 0, thal: 2 },
      negative: { age: 52, sex: 1, cp: 0, trestbps: 125, chol: 212, fbs: 0, restecg: 1, thalach: 168, exang: 0, oldpeak: 1.0, slope: 2, ca: 2, thal: 3 }
    };

    function loadSample(isPositive) {
      const s = isPositive ? samples.positive : samples.negative;
      Object.keys(s).forEach(k => { document.getElementById(k).value = s[k]; });
      document.getElementById('result').style.display = 'none';
    }

    document.getElementById('predictForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const payload = {
        age: parseInt(document.getElementById('age').value, 10),
        sex: parseInt(document.getElementById('sex').value, 10),
        cp: parseInt(document.getElementById('cp').value, 10),
        trestbps: parseInt(document.getElementById('trestbps').value, 10),
        chol: parseInt(document.getElementById('chol').value, 10),
        fbs: parseInt(document.getElementById('fbs').value, 10),
        restecg: parseInt(document.getElementById('restecg').value, 10),
        thalach: parseInt(document.getElementById('thalach').value, 10),
        exang: parseInt(document.getElementById('exang').value, 10),
        oldpeak: parseFloat(document.getElementById('oldpeak').value),
        slope: parseInt(document.getElementById('slope').value, 10),
        ca: parseInt(document.getElementById('ca').value, 10),
        thal: parseInt(document.getElementById('thal').value, 10)
      };

      const resBox = document.getElementById('result');
      resBox.style.display = 'block';
      resBox.className = '';
      resBox.textContent = 'Running prediction...';

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.heart_disease) {
          resBox.className = 'res-true';
          resBox.innerHTML = 'Prediction: <strong>Heart Disease Detected (heart_disease: true)</strong>';
        } else {
          resBox.className = 'res-false';
          resBox.innerHTML = 'Prediction: <strong>No Heart Disease Detected (heart_disease: false)</strong>';
        }
      } catch (err) {
        resBox.className = 'res-true';
        resBox.textContent = 'Error communicating with API: ' + err.message;
      }
    });
  </script>
</body>
</html>"""


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
