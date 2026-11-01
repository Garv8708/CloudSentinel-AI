from __future__ import annotations
import joblib
import pandas as pd
from src.config import MODEL_PATH

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python -m src.train_model --data data/network_traffic.csv"
        )
    return joblib.load(MODEL_PATH)

def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    X = df.copy()
    X.columns = [str(c).strip() for c in X.columns]
    # Remove label column if a labeled sample dataset is uploaded for demonstration.
    for col in ["Attack_type", "attack_type", "label", "Label", "class", "Class", "target", "Target"]:
        if col in X.columns:
            X = X.drop(columns=[col])
    expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    if expected:
        missing = sorted(set(expected) - set(X.columns))
        extra = sorted(set(X.columns) - set(expected))
        if missing:
            raise ValueError("Prediction CSV is missing required columns: " + ", ".join(missing[:20]))
        # Keep the same feature order as training; ignore unrelated extra columns.
        X = X[expected]
    pred = model.predict(X)
    result = df.copy()
    result["predicted_class"] = pred
    if hasattr(model, "predict_proba"):
        try:
            result["prediction_confidence"] = model.predict_proba(X).max(axis=1)
        except Exception:
            pass
    return result
