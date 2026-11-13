# Heart Disease Prediction API (FastAPI + Docker + Render)

A containerized FastAPI application that serves binary heart disease predictions (`heart_disease: true/false`) using a `RandomForestClassifier` trained on the Kaggle [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

## Live Deployment (Render)

- **Live Web UI**: [https://heart-disease-prediction-jg1t.onrender.com/](https://heart-disease-prediction-jg1t.onrender.com/)
- **Swagger UI (`/docs`)**: [https://heart-disease-prediction-jg1t.onrender.com/docs](https://heart-disease-prediction-jg1t.onrender.com/docs)
- **ReDoc (`/redoc`)**: [https://heart-disease-prediction-jg1t.onrender.com/redoc](https://heart-disease-prediction-jg1t.onrender.com/redoc)
- **Health Check (`/health`)**: [https://heart-disease-prediction-jg1t.onrender.com/health](https://heart-disease-prediction-jg1t.onrender.com/health)
- **Model Info (`/info`)**: [https://heart-disease-prediction-jg1t.onrender.com/info](https://heart-disease-prediction-jg1t.onrender.com/info)

## Project Structure

```text
heart-disease-prediction/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app & endpoints (/health, /info, /predict)
│   └── schemas.py           # Pydantic request/response schemas
├── model/
│   └── heart_model.joblib   # Trained RandomForestClassifier model artifact
├── heart.csv                # Heart Disease dataset
├── train_model.py           # Training script that produces model/heart_model.joblib
├── Dockerfile               # Container definition running uvicorn
├── docker-compose.yml       # Local Docker Compose service configuration
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## API Endpoints

| Method | Endpoint | Description | Sample Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health check endpoint | `{"status": "healthy"}` |
| `GET` | `/info` | Returns model type & feature list | `{"model_type": "RandomForestClassifier", "features": [...], "target_classes": {...}}` |
| `POST` | `/predict` | Predicts heart disease presence (`true`/`false`) | `{"heart_disease": true}` |

### Input Features (`POST /predict`)
- `age` (int): Age in years
- `sex` (int): Sex (`1` = male; `0` = female)
- `cp` (int): Chest pain type (`0`–`3`)
- `trestbps` (int): Resting blood pressure (in mm Hg)
- `chol` (int): Serum cholesterol in mg/dl
- `fbs` (int): Fasting blood sugar > 120 mg/dl (`1` = true; `0` = false)
- `restecg` (int): Resting electrocardiographic results (`0`–`2`)
- `thalach` (int): Maximum heart rate achieved
- `exang` (int): Exercise induced angina (`1` = yes; `0` = no)
- `oldpeak` (float): ST depression induced by exercise relative to rest
- `slope` (int): Slope of the peak exercise ST segment (`0`–`2`)
- `ca` (int): Number of major vessels (`0`–`4`) colored by fluoroscopy
- `thal` (int): Thalassemia (`0`–`3`)

---

## Running Locally with Docker Compose

### 1. Build the Docker Image
```bash
docker-compose build
```

### 2. Start the Container
```bash
docker-compose up
```

### 3. Access Swagger UI
Open your browser and navigate to:
- **Swagger UI**: http://127.0.0.1:8001/docs
- **ReDoc**: http://127.0.0.1:8001/redoc

---

## Testing Endpoints with `curl`

### Health Check (`GET /health`)
```bash
curl -X GET http://127.0.0.1:8001/health
```

### Model Info (`GET /info`)
```bash
curl -X GET http://127.0.0.1:8001/info
```

### Predict Heart Disease (`POST /predict`)
**Positive Example (`heart_disease: true`):**
```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "thal": 2
  }'
```

**Negative Example (`heart_disease: false`):**
```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "thal": 3
  }'
```

---

## Training the Model Locally (Without Docker)

```bash
pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --reload --port 8001
```

---

## Deploying to Render

1. Push this project directory to a GitHub repository.
2. Log in to [Render](https://render.com/) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Configure the service:
   - **Runtime / Language**: `Docker`
   - **Root Directory**: Leave blank if this project is at the repo root (or set to `heart-disease-prediction` if pushed inside a monorepo).
   - **Instance Type**: Free
5. Click **Create Web Service** and wait for the build to complete.
6. Test your live deployment at [https://heart-disease-prediction-jg1t.onrender.com/docs](https://heart-disease-prediction-jg1t.onrender.com/docs).
