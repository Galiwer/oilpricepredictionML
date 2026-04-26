from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import PreprocessingResult, run_preprocessing


@dataclass
class TrainingResult:
    model: xgb.XGBRegressor
    metrics: dict[str, float]
    preprocessing: PreprocessingResult
    report_paths: dict[str, str]


def build_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        early_stopping_rounds=20,
    )


def train_model(preprocessing: PreprocessingResult) -> TrainingResult:
    model = build_model()
    model.fit(
        preprocessing.X_train_scaled,
        preprocessing.y_train,
        eval_set=[
            (preprocessing.X_train_scaled, preprocessing.y_train),
            (preprocessing.X_test_scaled, preprocessing.y_test),
        ],
        verbose=False,
    )

    predictions = model.predict(preprocessing.X_test_scaled)
    metrics = {
        "mae": float(mean_absolute_error(preprocessing.y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(preprocessing.y_test, predictions))),
        "r2": float(r2_score(preprocessing.y_test, predictions)),
    }
    return TrainingResult(model=model, metrics=metrics, preprocessing=preprocessing, report_paths={})


def save_artifacts(training: TrainingResult, model_dir: str | Path) -> None:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(training.model, model_path / "xgboost_oil_model.joblib")
    joblib.dump(training.preprocessing.expected_columns, model_path / "expected_columns.joblib")
    joblib.dump(training.preprocessing.scaler, model_path / "scaler.joblib")


def generate_reports(training: TrainingResult, report_dir: str | Path) -> dict[str, str]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    y_test = training.preprocessing.y_test
    predictions = training.model.predict(training.preprocessing.X_test_scaled)

    scatter_file = report_path / "actual_vs_predicted_scatter.png"
    plt.figure()
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Scatter Plot")
    min_val = min(float(y_test.min()), float(predictions.min()))
    max_val = max(float(y_test.max()), float(predictions.max()))
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.savefig(scatter_file)
    plt.close()

    line_file = report_path / "actual_vs_predicted_line.png"
    results_df = pd.DataFrame(
        {
            "Index": range(len(y_test)),
            "Actual": y_test.values,
            "Predicted": predictions,
        }
    )
    plt.figure()
    plt.plot(results_df["Index"], results_df["Actual"], label="Actual Price")
    plt.plot(results_df["Index"], results_df["Predicted"], label="Predicted Price")
    plt.xlabel("Test Samples")
    plt.ylabel("Oil Price (USD per barrel)")
    plt.title("Actual vs Predicted Oil Prices")
    plt.legend()
    plt.savefig(line_file)
    plt.close()

    metrics_file = report_path / "metrics.json"
    metrics_file.write_text(json.dumps(training.metrics, indent=2), encoding="utf-8")

    return {
        "scatter_plot": str(scatter_file),
        "line_plot": str(line_file),
        "metrics_json": str(metrics_file),
    }


def run_training(
    data_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path | None = None,
) -> TrainingResult:
    preprocessing = run_preprocessing(data_path)
    training = train_model(preprocessing)
    save_artifacts(training, model_dir)
    if report_dir is not None:
        training.report_paths = generate_reports(training, report_dir)
    return training


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "Middle_East_Economic_Data_1990_2024_with_Oil.csv"
    models_path = project_root / "models"
    reports_path = project_root / "reports"
    result = run_training(dataset_path, models_path, reports_path)
    print("Training complete")
    print(
        f"MAE: ${result.metrics['mae']:.2f} | RMSE: ${result.metrics['rmse']:.2f} | R2: {result.metrics['r2']:.4f}"
    )
    print(f"Reports: {result.report_paths}")