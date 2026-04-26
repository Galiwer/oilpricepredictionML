from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = [
    "GDP_growth_annual_pct",
    "Inflation_consumer_prices_annual_pct",
    "Exports_pct_GDP",
    "Unemployment_total_pct",
    "Oil_Price_Lag1",
]
DEFAULT_TARGET = "Future_Oil_Price"


@dataclass
class PreprocessingResult:
    X_train_scaled: object
    X_test_scaled: object
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    expected_columns: list[str]
    processed_df: pd.DataFrame


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Country_Code" in df.columns:
        df = df.drop(columns=["Country_Code"])

    df = df.sort_values(by=["Country", "Year"])
    cols_to_fill = df.columns.drop(["Country", "Year"])

    for col in cols_to_fill:
        df[col] = df.groupby("Country")[col].transform(
            lambda x: x.fillna(x.rolling(window=10, min_periods=1).median())
        )

    df[cols_to_fill] = df.groupby("Country")[cols_to_fill].bfill()
    df[cols_to_fill] = df.groupby("Country")[cols_to_fill].ffill()

    df["Future_Oil_Price"] = df.groupby("Country")["Brent_Oil_Price_USD_per_barrel"].shift(-1)
    df["Oil_Price_Lag1"] = df.groupby("Country")["Brent_Oil_Price_USD_per_barrel"].shift(1)
    df = df.dropna(subset=["Future_Oil_Price", "Oil_Price_Lag1"])

    return df


def split_scale_dataset(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessingResult:
    selected_features = features or DEFAULT_FEATURES
    X = df[selected_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return PreprocessingResult(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        expected_columns=X_train.columns.tolist(),
        processed_df=df,
    )


def run_preprocessing(
    data_path: str | Path,
    features: list[str] | None = None,
    target: str = DEFAULT_TARGET,
) -> PreprocessingResult:
    raw_df = load_raw_data(data_path)
    processed_df = preprocess_dataframe(raw_df)
    return split_scale_dataset(processed_df, features=features, target=target)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "Middle_East_Economic_Data_1990_2024_with_Oil.csv"
    preprocessing_result = run_preprocessing(dataset_path)
    print("Preprocessing complete")
    print(f"Train shape: {preprocessing_result.X_train_scaled.shape}")
    print(f"Test shape: {preprocessing_result.X_test_scaled.shape}")