from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import run_full_pipeline
from src.predict import load_artifacts, predict_from_features

app = FastAPI(title="Oil Price Prediction API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Middle_East_Economic_Data_1990_2024_with_Oil.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"


class PredictionInput(BaseModel):
    GDP_growth_annual_pct: float
    Inflation_consumer_prices_annual_pct: float
    Exports_pct_GDP: float
    Unemployment_total_pct: float
    Oil_Price_Lag1: float


@app.get("/")
def root() -> dict:
    return {
        "service": "Oil Price Prediction API",
        "docs": "/docs",
        "routes": {
            "GET /health": "Health check",
            "POST /train": "Train model and generate reports",
            "POST /predict": "Predict future oil price",
        },
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/train")
def train_pipeline() -> dict:
    try:
        result = run_full_pipeline(data_path=DATA_PATH, model_dir=MODEL_DIR, report_dir=REPORT_DIR)
        return {
            "message": "Training completed successfully",
            "metrics": result["metrics"],
            "reports": result["reports"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@app.post("/predict")
def predict_oil_price(payload: PredictionInput) -> dict:
    try:
        model, scaler, expected_columns = load_artifacts(MODEL_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail="Model artifacts not found. Run POST /train first.",
        ) from exc

    prediction = predict_from_features(
        input_data=payload.model_dump(),
        model=model,
        scaler=scaler,
        expected_columns=expected_columns,
    )
    return {"predicted_future_oil_price": prediction}