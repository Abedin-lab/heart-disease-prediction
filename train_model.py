"""Script to train and save the Heart Disease prediction model."""

import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "heart.csv"
    if not data_path.exists():
        # Fallback to parent directory if running before copy
        data_path = base_dir.parent / "heart.csv"

    df = pd.read_csv(data_path)
    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    model_dir = base_dir / "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "heart_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved trained model to: {model_path}")


if __name__ == "__main__":
    main()
