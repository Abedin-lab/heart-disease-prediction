FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset, training script, trained model, and application code
COPY heart.csv train_model.py ./
COPY model/ ./model/
COPY app/ ./app/

# Ensure model artifact is compatible with the container's scikit-learn version
RUN python train_model.py

EXPOSE 8000

# Run FastAPI via uvicorn (respects Render's $PORT if provided, defaults to 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
