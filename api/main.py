from __future__ import annotations
from io import BytesIO
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from src.config import MAX_UPLOAD_MB
from src.predict import predict_frame

app = FastAPI(
    title="CloudSentinel AI API",
    description="Educational API for classifying network traffic records from a CSV dataset.",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok", "application": "CloudSentinel AI"}

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB.")
    try:
        df = pd.read_csv(BytesIO(content), low_memory=False)
        if df.empty:
            raise ValueError("CSV has no rows.")
        result = predict_frame(df)
        return {
            "rows_analyzed": len(result),
            "predictions": result.head(1000).to_dict(orient="records"),
            "note": "Educational predictions only; suspicious predictions are not verified attacks."
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process CSV: {exc}")
