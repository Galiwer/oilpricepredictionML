# Oil Price Prediction Pipeline

This project now uses a function-based ML pipeline that can be imported directly from FastAPI, instead of running subprocess commands for each script.

## Project Flow

1. `src/preprocessing.py`
	- `run_preprocessing(...)` loads data, engineers features, splits, and scales.
2. `src/train.py`
	- `run_training(...)` trains XGBoost and saves artifacts to `models/`.
3. `src/predict.py`
	- `predict_with_saved_artifacts(...)` loads artifacts and predicts from feature input.
4. `src/pipeline.py`
	- `run_full_pipeline(...)` orchestrates train + optional sample prediction.

## FastAPI Endpoints

- `GET /health` - health check.
- `POST /train` - runs preprocessing + training and stores model artifacts.
  - Also generates reports in `reports/`:
    - `actual_vs_predicted_scatter.png`
    - `actual_vs_predicted_line.png`
    - `metrics.json`
- `POST /predict` - predicts next oil price from engineered features.

Request body for `POST /predict`:

```json
{
  "GDP_growth_annual_pct": 3.5,
  "Inflation_consumer_prices_annual_pct": 2.5,
  "Exports_pct_GDP": 35.2,
  "Unemployment_total_pct": 5.1,
  "Oil_Price_Lag1": 82.5
}
```

## Run API

```bash
uvicorn api.main:app --reload
```

## Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard includes:

- Dataset exploration by country and year.
- An interactive prediction form using the saved model artifacts.
- Saved training reports and metrics from `reports/`.

## Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

## Run With Docker

### Build image

```bash
docker build -t oilpriceprediction-api .
```

### Run container

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/data:/app/data:ro" \
  oilpriceprediction-api
```

### Run with Docker Compose

```bash
docker compose up --build
```

This starts both the FastAPI service on `http://127.0.0.1:8000` and the Streamlit dashboard on `http://127.0.0.1:8501`.
