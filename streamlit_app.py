from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.pipeline import run_full_pipeline
from src.predict import load_artifacts, predict_from_features
from src.preprocessing import load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Middle_East_Economic_Data_1990_2024_with_Oil.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURE_FIELDS = [
    ("GDP_growth_annual_pct", 3.5, -20.0, 20.0, 0.1),
    ("Inflation_consumer_prices_annual_pct", 2.5, -10.0, 50.0, 0.1),
    ("Exports_pct_GDP", 35.2, 0.0, 200.0, 0.1),
    ("Unemployment_total_pct", 5.1, 0.0, 50.0, 0.1),
    ("Oil_Price_Lag1", 82.5, 0.0, 200.0, 0.1),
]


if get_script_run_ctx(suppress_warning=True) is None:
    raise SystemExit("Run this app with: streamlit run streamlit_app.py")


st.set_page_config(page_title="Oil Price Dashboard", page_icon="📈", layout="wide")


@st.cache_data

def load_dataset() -> pd.DataFrame:
    return load_raw_data(DATA_PATH)


@st.cache_resource

def load_saved_artifacts():
    return load_artifacts(MODEL_DIR)


@st.cache_data

def load_metrics() -> dict:
    metrics_path = REPORT_DIR / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


def run_training_pipeline() -> dict:
    return run_full_pipeline(DATA_PATH, MODEL_DIR, REPORT_DIR)


st.title("Oil Price Prediction Dashboard")
st.caption("Explore the dataset, retrain the model, and generate future oil price predictions.")

with st.sidebar:
    st.header("Controls")
    st.write("Use the buttons below to refresh model artifacts or explore the saved reports.")
    if st.button("Retrain model and regenerate reports", width="stretch"):
        with st.spinner("Training model and generating reports..."):
            result = run_training_pipeline()
            st.success("Training complete")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    if DATA_PATH.exists():
        st.success("Dataset found")
    else:
        st.error("Dataset not found")

    if (MODEL_DIR / "xgboost_oil_model.joblib").exists():
        st.success("Model artifacts available")
    else:
        st.warning("Train the model first")

    st.link_button("Open API docs", "http://127.0.0.1:8000/docs")


tab_overview, tab_predict, tab_reports = st.tabs(["Overview", "Prediction", "Reports"])

with tab_overview:
    df = load_dataset()
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Countries", f"{df['Country'].nunique():,}")
    col3.metric("Years", f"{df['Year'].nunique():,}")

    left, right = st.columns([1, 1])
    with left:
        country = st.selectbox("Select country", sorted(df["Country"].dropna().unique()))
        country_df = df[df["Country"] == country].sort_values("Year")
        st.subheader(f"Oil price trend: {country}")
        st.line_chart(country_df.set_index("Year")["Brent_Oil_Price_USD_per_barrel"])
        st.dataframe(
            country_df[["Year", "Brent_Oil_Price_USD_per_barrel", "GDP_growth_annual_pct", "Inflation_consumer_prices_annual_pct"]].tail(10),
            width="stretch",
        )

    with right:
        st.subheader("Dataset snapshot")
        st.dataframe(df.head(15), width="stretch")
        st.subheader("Feature distribution")
        feature_choice = st.selectbox(
            "Choose a numeric feature",
            [
                "GDP_growth_annual_pct",
                "Inflation_consumer_prices_annual_pct",
                "Exports_pct_GDP",
                "Unemployment_total_pct",
                "Brent_Oil_Price_USD_per_barrel",
            ],
        )
        st.bar_chart(df.groupby("Country")[feature_choice].mean().sort_values(ascending=False).head(10))

with tab_predict:
    st.subheader("Predict future oil price")
    try:
        model, scaler, expected_columns = load_saved_artifacts()
    except Exception:
        model = scaler = expected_columns = None
        st.warning("Train the model first so the prediction form can use saved artifacts.")

    with st.form("prediction_form"):
        values: dict[str, float] = {}
        for field_name, default_value, min_value, max_value, step in FEATURE_FIELDS:
            values[field_name] = st.number_input(
                field_name,
                value=float(default_value),
                min_value=float(min_value),
                max_value=float(max_value),
                step=float(step),
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        if model is None:
            st.error("No saved model artifacts found. Run training first.")
        else:
            prediction = predict_from_features(values, model, scaler, expected_columns)
            st.success(f"Predicted future oil price: ${prediction:.2f}")

    st.info("The form uses the engineered features that the model was trained on.")

with tab_reports:
    st.subheader("Training reports")
    metrics = load_metrics()
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"${metrics.get('mae', 0):.2f}")
        c2.metric("RMSE", f"${metrics.get('rmse', 0):.2f}")
        c3.metric("R2", f"{metrics.get('r2', 0):.4f}")
    else:
        st.warning("No metrics found. Train the model to generate reports.")

    scatter_path = REPORT_DIR / "actual_vs_predicted_scatter.png"
    line_path = REPORT_DIR / "actual_vs_predicted_line.png"

    image_col1, image_col2 = st.columns(2)
    with image_col1:
        if scatter_path.exists():
            st.image(str(scatter_path), caption="Actual vs Predicted Scatter", width="stretch")
        else:
            st.info("Scatter plot not found yet.")
    with image_col2:
        if line_path.exists():
            st.image(str(line_path), caption="Actual vs Predicted Line", width="stretch")
        else:
            st.info("Line chart not found yet.")

    metrics_path = REPORT_DIR / "metrics.json"
    if metrics_path.exists():
        st.code(metrics_path.read_text(encoding="utf-8"), language="json")
