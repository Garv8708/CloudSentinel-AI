from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from src.config import ARTIFACTS_DIR, MODEL_PATH, METADATA_PATH
from src.data_utils import split_features_target

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ], remainder="drop")
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    return Pipeline([("preprocess", prep), ("classifier", model)])

def main():
    parser = argparse.ArgumentParser(description="Train CloudSentinel traffic classifier")
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    X, y, target = split_features_target(df)
    if y.nunique() < 2:
        raise SystemExit("Need at least two target classes to train a classifier.")
    # Stratification is possible only when every class has at least two records.
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    labels = sorted(y.unique().tolist())
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=labels)
    metrics = {
        "target_column": target,
        "rows_after_cleaning": int(len(X)),
        "feature_columns": list(X.columns),
        "classes": labels,
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "test_rows": int(len(X_test)),
        "note": "Metrics are from a random held-out split; duplicates or related records can cause optimistic results. Consider group/time-aware validation when applicable."
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")
    print(f"Accuracy on held-out split: {metrics['accuracy']:.4f}")
    print(classification_report(y_test, pred, zero_division=0))

if __name__ == "__main__":
    main()
