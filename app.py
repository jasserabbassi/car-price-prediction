"""Streamlit dashboard for the 9-model ensemble car price predictor."""

from __future__ import annotations

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    CATEGORICAL_FEATURES,
    CURRENT_YEAR,
    DATA_FILE,
    ENCODER_FILE,
    MODEL_ADABOOST,
    MODEL_ENSEMBLE,
    MODEL_GRADIENT_BOOSTING,
    MODEL_LASSO,
    MODEL_LINEAR,
    MODEL_MLP,
    MODEL_RANDOM_FOREST,
    MODEL_RIDGE,
    MODEL_SVR,
    MODEL_XGBOOST,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    SCALER_FILE,
)
from quality_report import generate_quality_report
from utils import preprocess_input_data
# WeightedEnsemble import lets pickle.load resolve the class.
from train_model import WeightedEnsemble  # noqa: F401

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATHS = {
    "Linear Regression": MODEL_LINEAR,
    "Ridge Regression": MODEL_RIDGE,
    "Lasso Regression": MODEL_LASSO,
    "Support Vector Regression": MODEL_SVR,
    "Random Forest": MODEL_RANDOM_FOREST,
    "Gradient Boosting": MODEL_GRADIENT_BOOSTING,
    "XGBoost": MODEL_XGBOOST,
    "AdaBoost": MODEL_ADABOOST,
    "MLP": MODEL_MLP,
}


@st.cache_resource
def load_models():
    base_models: dict[str, object] = {}
    for name, path in MODEL_PATHS.items():
        if Path(path).exists():
            try:
                with open(path, "rb") as f:
                    base_models[name] = pickle.load(f)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not load {name}: {exc}")
        else:
            st.info(f"{name} not trained yet. Run `python train_model.py` first.")

    ensemble = None
    if Path(MODEL_ENSEMBLE).exists():
        with open(MODEL_ENSEMBLE, "rb") as f:
            ensemble = pickle.load(f)

    scaler = None
    if Path(SCALER_FILE).exists():
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)

    encoder = None
    if Path(ENCODER_FILE).exists():
        with open(ENCODER_FILE, "rb") as f:
            encoder = pickle.load(f)

    return base_models, ensemble, scaler, encoder


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_artifacts() -> dict:
    out = {}
    for fname in (
        "ensemble_weights.json",
        "all_metrics.json",
        "per_fold_r2.json",
        "wilcoxon_results.json",
    ):
        path = MODELS_DIR / fname
        out[fname] = json.loads(path.read_text()) if path.exists() else None

    shap_path = MODELS_DIR / "shap_summary.png"
    out["shap_summary_png"] = shap_path if shap_path.exists() else None

    shap_csv = MODELS_DIR / "shap_importance.csv"
    out["shap_importance"] = pd.read_csv(shap_csv) if shap_csv.exists() else None

    return out


# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #

def show_home_page(metrics: dict | None) -> None:
    st.markdown("## Welcome to the Car Price Prediction System")

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            """
            ### Highlights
            - 9 ML models (Linear / Ridge / Lasso / SVR / RF / GB / XGBoost / AdaBoost / MLP)
            - Performance-weighted ensemble with weights ∝ validation R²
            - 5-fold cross-validation + Wilcoxon significance test
            - SHAP explainability and an interactive dashboard
            """
        )
    with col2:
        st.success(
            """
            ### Quick Start
            1. Open **Price Predictor**
            2. Fill in the car details
            3. Click **Predict**
            4. Inspect every individual prediction + the ensemble
            """
        )

    st.markdown("---")
    cols = st.columns(4)
    cols[0].metric("Base models", "9")
    if metrics and "Weighted Ensemble" in metrics:
        cols[1].metric("Ensemble Test R²", f"{metrics['Weighted Ensemble']['r2']:.4f}")
        cols[2].metric("Ensemble RMSE", f"${metrics['Weighted Ensemble']['rmse']:,.0f}")
        cols[3].metric("Ensemble MAE", f"${metrics['Weighted Ensemble']['mae']:,.0f}")
    else:
        cols[1].metric("Ensemble Test R²", "—")
        cols[2].metric("Ensemble RMSE", "—")
        cols[3].metric("Ensemble MAE", "—")


def _safe_options(encoder, col: str, fallback: list[str]) -> list[str]:
    if encoder and col in encoder:
        try:
            return sorted([str(x) for x in encoder[col].classes_])
        except Exception:  # noqa: BLE001
            return fallback
    return fallback


def show_predictor_page(base_models, ensemble, scaler, encoder, weights: dict | None) -> None:
    st.markdown("## Car Price Predictor")

    if not base_models or scaler is None or encoder is None:
        st.error("Models or preprocessors are missing. Run `python train_model.py` first.")
        return

    brand_options = _safe_options(encoder, "Brand", ["Toyota", "Honda", "BMW"])
    fuel_options = _safe_options(encoder, "Fuel Type", ["Petrol", "Diesel", "Hybrid"])
    transmission_options = _safe_options(encoder, "Transmission", ["Manual", "Automatic"])
    condition_options = _safe_options(encoder, "Condition", ["New", "Used"])
    model_options = _safe_options(encoder, "Model", ["Corolla"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Car Details")
        col_a, col_b = st.columns(2)
        with col_a:
            brand = st.selectbox("Brand", brand_options)
            year = st.slider("Year", 1990, CURRENT_YEAR, 2018)
            engine_size = st.number_input("Engine Size (L)", 0.5, 8.0, 2.0, 0.1)
            fuel_type = st.selectbox("Fuel Type", fuel_options)
        with col_b:
            model_value = st.selectbox(
                "Model", model_options[: min(len(model_options), 200)]
            )
            mileage = st.number_input("Mileage (mi)", 0, 500_000, 60_000)
            transmission = st.selectbox("Transmission", transmission_options)
            condition = st.selectbox("Condition", condition_options)

    with col2:
        st.markdown("### Summary")
        st.write(f"**Brand**: {brand}")
        st.write(f"**Model**: {model_value}")
        st.write(f"**Year**: {year}  (age = {CURRENT_YEAR - year})")
        st.write(f"**Engine**: {engine_size} L  •  **Fuel**: {fuel_type}")
        st.write(f"**Transmission**: {transmission}  •  **Condition**: {condition}")

    if not st.button("Predict", type="primary", use_container_width=True):
        return

    payload = {
        "Brand": brand,
        "Model": model_value,
        "Year": year,
        "Engine Size": engine_size,
        "Mileage": mileage,
        "Fuel Type": fuel_type,
        "Transmission": transmission,
        "Condition": condition,
    }

    try:
        X = preprocess_input_data(payload, encoder, scaler)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not prepare input: {exc}")
        return

    individual: dict[str, float] = {}
    for name, model in base_models.items():
        try:
            pred = float(np.asarray(model.predict(X)).reshape(-1)[0])
            individual[name] = max(0.0, pred)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{name} failed to predict: {exc}")

    if not individual:
        st.error("No model produced a prediction.")
        return

    # Weighted ensemble.
    if ensemble is not None:
        ensemble_pred = float(np.asarray(ensemble.predict(X)).reshape(-1)[0])
        ensemble_label = "Weighted Ensemble (production)"
    else:
        ensemble_pred = float(np.mean(list(individual.values())))
        ensemble_label = "Simple mean of predictions"

    arr = np.fromiter(individual.values(), dtype=float)

    cols = st.columns(4)
    cols[0].metric(ensemble_label, f"${ensemble_pred:,.0f}")
    cols[1].metric("Min individual", f"${arr.min():,.0f}")
    cols[2].metric("Max individual", f"${arr.max():,.0f}")
    cols[3].metric("Models used", f"{len(individual)}")

    pred_df = pd.DataFrame(
        {
            "Model": list(individual.keys()),
            "Prediction": list(individual.values()),
            "Weight": [(weights or {}).get(name, 0.0) for name in individual.keys()],
        }
    ).sort_values("Prediction")

    st.markdown("### Predictions per model")
    fig = px.bar(
        pred_df,
        x="Prediction",
        y="Model",
        color="Weight",
        orientation="h",
        title="Individual model predictions (color = ensemble weight)",
        color_continuous_scale="Viridis",
    )
    fig.add_vline(
        x=ensemble_pred,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Ensemble = ${ensemble_pred:,.0f}",
    )
    st.plotly_chart(fig, use_container_width=True)

    if weights:
        st.markdown("### Ensemble weights (∝ validation R²)")
        wdf = (
            pd.DataFrame({"Model": list(weights.keys()), "Weight": list(weights.values())})
            .sort_values("Weight", ascending=False)
        )
        st.dataframe(wdf, use_container_width=False, hide_index=True)

    # Persist to history.
    history_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": payload,
        "predictions": individual,
        "ensemble": ensemble_pred,
    }
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    st.session_state.prediction_history.append(history_payload)


def show_eda_page(df: pd.DataFrame) -> None:
    st.markdown("## Data Analysis")
    tabs = st.tabs(["Overview", "Distributions", "Market Analysis", "Correlations"])

    with tabs[0]:
        st.subheader("Dataset overview")
        cols = st.columns(3)
        cols[0].metric("Rows", f"{len(df):,}")
        cols[1].metric("Features", len(df.columns))
        cols[2].metric("Missing cells", int(df.isnull().sum().sum()))
        st.dataframe(df.head(15), use_container_width=True)

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Price", nbins=60, title="Price distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x="Mileage", nbins=60, title="Mileage distribution")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            top_brands = df.groupby("Brand")["Price"].mean().sort_values(ascending=False).head(15)
            st.plotly_chart(
                px.bar(top_brands, title="Average price by brand (top 15)"),
                use_container_width=True,
            )
        with col2:
            year_price = df.groupby("Year")["Price"].mean()
            st.plotly_chart(
                px.line(year_price, title="Average price by year"),
                use_container_width=True,
            )

    with tabs[3]:
        numeric_df = df.select_dtypes(include=[np.number])
        st.plotly_chart(
            px.imshow(
                numeric_df.corr(),
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Correlation heatmap",
            ),
            use_container_width=True,
        )


def show_model_details_page(metrics: dict | None, weights: dict | None, shap_png: Path | None,
                            shap_imp: pd.DataFrame | None) -> None:
    st.markdown("## Model Details & Performance")

    if metrics:
        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": name,
                "R²": m.get("r2", float("nan")),
                "RMSE": m.get("rmse", float("nan")),
                "MAE": m.get("mae", float("nan")),
                "MAPE %": m.get("mape", float("nan")),
            })
        df = pd.DataFrame(rows).sort_values("R²", ascending=False)
        st.subheader("Held-out test set metrics")
        st.dataframe(
            df.style.format({"R²": "{:.4f}", "RMSE": "${:,.0f}",
                              "MAE": "${:,.0f}", "MAPE %": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(
            px.bar(df, x="Model", y="R²", title="Test R² per model"),
            use_container_width=True,
        )

    if weights:
        st.subheader("Performance-weighted ensemble — weights")
        wdf = (
            pd.DataFrame({"Model": list(weights.keys()), "Weight": list(weights.values())})
            .sort_values("Weight", ascending=False)
        )
        st.plotly_chart(
            px.bar(wdf, x="Weight", y="Model", orientation="h", title="Ensemble weights"),
            use_container_width=True,
        )

    if shap_imp is not None:
        st.subheader("SHAP feature importance (XGBoost)")
        st.dataframe(shap_imp, use_container_width=False, hide_index=True)
    if shap_png is not None:
        st.image(str(shap_png), caption="SHAP summary — XGBoost", use_column_width=True)


def show_statistical_tests_page(per_fold: dict | None, wilcoxon: list | None) -> None:
    st.markdown("## Statistical Tests")

    if per_fold:
        st.subheader("Per-fold R² (5-fold CV)")
        df_pf = pd.DataFrame(per_fold)
        df_pf.index = [f"Fold {i+1}" for i in range(len(df_pf))]
        st.dataframe(df_pf.style.format("{:.4f}"), use_container_width=True)

        means = df_pf.mean().sort_values(ascending=False)
        stds = df_pf.std()

        summary = pd.DataFrame({"CV mean": means, "CV std": stds.reindex(means.index)}).reset_index()
        summary.columns = ["Model", "CV mean", "CV std"]
        st.plotly_chart(
            px.bar(summary, x="Model", y="CV mean", error_y="CV std",
                   title="5-fold CV R² (with standard deviation)"),
            use_container_width=True,
        )

    if wilcoxon:
        st.subheader("Wilcoxon signed-rank test (Ensemble vs each base model)")
        st.markdown(
            "One-sided alternative: ensemble per-fold R² is greater than the base model's. "
            "α = 0.05 throughout."
        )
        wdf = pd.DataFrame(wilcoxon)
        st.dataframe(
            wdf[["Comparison", "Ensemble mean R²", "Base mean R²", "Mean Δ", "W",
                  "p-value (one-sided)", "Significant @ α=0.05"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No Wilcoxon results yet. Run `python wilcoxon_test.py` after training.")


def show_data_quality_page(df: pd.DataFrame) -> None:
    st.markdown("## Data Quality Report")
    report = generate_quality_report(df)
    st.json(report)


def show_history_page() -> None:
    st.markdown("## Prediction History")
    history = st.session_state.get("prediction_history", [])
    if not history:
        st.info("No predictions yet. Make one from the Price Predictor page.")
        return
    rows = []
    for h in history:
        rows.append({
            "Time": h["timestamp"],
            "Brand": h["input"]["Brand"],
            "Model": h["input"]["Model"],
            "Year": h["input"]["Year"],
            "Mileage": h["input"]["Mileage"],
            "Ensemble Price": h.get("ensemble", float("nan")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    st.title("Car Price Prediction System")

    page = st.sidebar.radio(
        "Select Page",
        [
            "Home",
            "Price Predictor",
            "Data Analysis",
            "Model Details",
            "Statistical Tests",
            "Data Quality Report",
            "Prediction History",
        ],
    )

    base_models, ensemble, scaler, encoder = load_models()
    df = load_dataset()
    artifacts = load_artifacts()
    weights = artifacts["ensemble_weights.json"]
    metrics = artifacts["all_metrics.json"]

    if page == "Home":
        show_home_page(metrics)
    elif page == "Price Predictor":
        show_predictor_page(base_models, ensemble, scaler, encoder, weights)
    elif page == "Data Analysis":
        show_eda_page(df)
    elif page == "Model Details":
        show_model_details_page(
            metrics, weights, artifacts["shap_summary_png"], artifacts["shap_importance"]
        )
    elif page == "Statistical Tests":
        show_statistical_tests_page(
            artifacts["per_fold_r2.json"], artifacts["wilcoxon_results.json"]
        )
    elif page == "Data Quality Report":
        show_data_quality_page(df)
    elif page == "Prediction History":
        show_history_page()


if __name__ == "__main__":
    main()
