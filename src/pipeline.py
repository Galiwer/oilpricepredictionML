from pathlib import Path
from typing import NotRequired, TypedDict

from src.predict import load_artifacts, predict_from_features
from src.train import run_training


class PipelineResult(TypedDict):
    metrics: dict[str, float]
    artifacts_dir: str
    reports: dict[str, str]
    sample_prediction: NotRequired[float]


def run_full_pipeline(
    data_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path | None = None,
    sample_input: dict | None = None,
) -> PipelineResult:
    training = run_training(data_path=data_path, model_dir=model_dir, report_dir=report_dir)

    response: PipelineResult = {
        "metrics": training.metrics,
        "artifacts_dir": str(Path(model_dir)),
        "reports": training.report_paths,
    }

    if sample_input:
        model, scaler, expected_columns = load_artifacts(model_dir)
        response["sample_prediction"] = predict_from_features(
            sample_input,
            model=model,
            scaler=scaler,
            expected_columns=expected_columns,
        )

    return response


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "Middle_East_Economic_Data_1990_2024_with_Oil.csv"
    models_path = project_root / "models"
    reports_path = project_root / "reports"

    payload = {
        "GDP_growth_annual_pct": 3.5,
        "Inflation_consumer_prices_annual_pct": 2.5,
        "Exports_pct_GDP": 35.2,
        "Unemployment_total_pct": 5.1,
        "Oil_Price_Lag1": 82.5,
    }

    result = run_full_pipeline(dataset_path, models_path, reports_path, sample_input=payload)
    print(result)
