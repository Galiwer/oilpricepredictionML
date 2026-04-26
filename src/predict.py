from pathlib import Path

import joblib
import pandas as pd


def load_artifacts(model_dir: str | Path):
    artifacts_path = Path(model_dir)
    model = joblib.load(artifacts_path / "xgboost_oil_model.joblib")
    scaler = joblib.load(artifacts_path / "scaler.joblib")
    expected_columns = joblib.load(artifacts_path / "expected_columns.joblib")
    return model, scaler, expected_columns


def prepare_features(input_data: dict, expected_columns: list[str]) -> pd.DataFrame:
    feature_df = pd.DataFrame([input_data])
    feature_df = feature_df.reindex(columns=expected_columns, fill_value=0)
    return feature_df


def predict_from_features(
    input_data: dict,
    model,
    scaler,
    expected_columns: list[str],
) -> float:
    feature_df = prepare_features(input_data, expected_columns)
    scaled = scaler.transform(feature_df)
    prediction = model.predict(scaled)
    return float(prediction[0])


def predict_with_saved_artifacts(input_data: dict, model_dir: str | Path) -> float:
    model, scaler, expected_columns = load_artifacts(model_dir)
    return predict_from_features(input_data, model, scaler, expected_columns)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sample_input = {
        "GDP_growth_annual_pct": 3.5,
        "Inflation_consumer_prices_annual_pct": 2.5,
        "Exports_pct_GDP": 35.2,
        "Unemployment_total_pct": 5.1,
        "Oil_Price_Lag1": 82.5,
    }
    output = predict_with_saved_artifacts(sample_input, project_root / "models")
    print(f"Predicted oil price: ${output:.2f}")