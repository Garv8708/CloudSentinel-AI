from __future__ import annotations
import pandas as pd
from src.config import CANDIDATE_TARGET_COLUMNS, TARGET_COLUMN

def find_target_column(df: pd.DataFrame) -> str:
    if TARGET_COLUMN and TARGET_COLUMN in df.columns:
        return TARGET_COLUMN
    for name in CANDIDATE_TARGET_COLUMNS:
        if name in df.columns:
            return name
    raise ValueError(
        "Could not find the target/label column. Available columns: "
        + ", ".join(map(str, df.columns))
        + ". Set TARGET_COLUMN in src/config.py to the correct column name."
    )

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop exact duplicate records; keep remaining values for sklearn preprocessing.
    df = df.drop_duplicates()
    # Drop completely empty columns.
    df = df.dropna(axis=1, how="all")
    return df

def split_features_target(df: pd.DataFrame):
    df = clean_frame(df)
    target = find_target_column(df)
    df = df.dropna(subset=[target])
    y = df[target].astype(str)
    X = df.drop(columns=[target])
    if X.shape[1] == 0:
        raise ValueError("Dataset has no feature columns after removing the target.")
    # Avoid accidental direct label leakage from obvious identifier columns.
    id_like = [c for c in X.columns if str(c).strip().lower() in {"id", "row_id", "record_id"}]
    if id_like:
        X = X.drop(columns=id_like)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")
    return X, y, target
