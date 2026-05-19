# -*- coding: utf-8 -*-
"""Streamlit dashboard for the 9-model ensemble car price predictor.

A single-file Streamlit app structured by page. Every page is a function
named `show_*_page()`; the sidebar selects which one runs.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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
# Importing train_model registers WeightedEnsemble on the running __main__,
# so old pickles still load.
from train_model import WeightedEnsemble  # noqa: F401

warnings.filterwarnings("ignore")

GITHUB_URL = "https://github.com/jasserabbassi/car-price-prediction"
APP_VERSION = "v2.0"
ACCENT = "#d4f542"
SUCCESS = "#d4f542"
WARN = "#6b6b60"
DANGER = "#2a2a26"
HISTORY_FILE = Path("prediction_history.json")

# Global Plotly template — industrial editorial aesthetic
pio.templates["industrial"] = {
    "layout": {
        "paper_bgcolor": "#0a0a08",
        "plot_bgcolor": "#111110",
        "font": {"color": "#f0ede6", "family": "DM Mono, monospace", "size": 11},
        "xaxis": {
            "gridcolor": "#2a2a26",
            "zerolinecolor": "#2a2a26",
            "linecolor": "#2a2a26",
            "tickfont": {"family": "DM Mono, monospace", "color": "#6b6b60", "size": 10},
            "title": {"font": {"family": "Syne, sans-serif", "color": "#f0ede6", "size": 12}},
        },
        "yaxis": {
            "gridcolor": "#2a2a26",
            "zerolinecolor": "#2a2a26",
            "linecolor": "#2a2a26",
            "tickfont": {"family": "DM Mono, monospace", "color": "#6b6b60", "size": 10},
            "title": {"font": {"family": "Syne, sans-serif", "color": "#f0ede6", "size": 12}},
        },
        "colorway": ["#d4f542", "#f0ede6", "#6b6b60", "#2a2a26"],
        "hoverlabel": {
            "bgcolor": "#111110",
            "font": {"color": "#f0ede6", "family": "DM Mono, monospace"},
            "bordercolor": "#2a2a26",
        },
        "legend": {"font": {"family": "DM Mono, monospace", "color": "#6b6b60", "size": 10}},
        "title": {"font": {"family": "Syne, sans-serif", "color": "#f0ede6", "size": 14}},
        "coloraxis": {
            "colorbar": {
                "tickfont": {"family": "DM Mono, monospace", "color": "#6b6b60", "size": 9},
                "title": {"font": {"family": "Syne, sans-serif", "color": "#f0ede6", "size": 11}},
            }
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    },
    "data": {
        "scatter": [{"marker": {"line": {"color": "#2a2a26", "width": 0.5}}}],
        "bar": [{"marker": {"line": {"color": "#2a2a26", "width": 0.5}}}],
    },
}
pio.templates.default = "industrial"

st.set_page_config(
    page_title="Car Price Predictor — 9-Model Ensemble",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": GITHUB_URL,
        "Report a bug": f"{GITHUB_URL}/issues",
        "About": "9-model heterogeneous ensemble for used-car price prediction. "
                 "Master's thesis project (2025).",
    },
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300..800;1,9..144,300..800&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

* {{ font-family: 'DM Mono', monospace !important; }}
.material-icons, [class*="material-icons"], [class*="material-symbols"], i.icon, span.icon, svg, svg * {{ font-family: 'Material Icons Outlined', 'Material Icons', 'Material Symbols Outlined', sans-serif !important; }}
html, body, .stApp {{ background: #0a0a08 !important; color: #f0ede6; }}

/* ---- layered industrial bg ---- */
.stApp::before {{
    content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background:
        repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(42,42,38,0.03) 40px, rgba(42,42,38,0.03) 41px),
        repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(42,42,38,0.03) 40px, rgba(42,42,38,0.03) 41px);
    pointer-events: none; z-index: 0;
}}

.block-container {{ position: relative; z-index: 1; padding-top: 1.2rem !important; padding-bottom: 3rem !important; max-width: 1300px !important; }}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {{ background: #0d0d0b !important; border-right: 1px solid #2a2a26 !important; }}
button[data-testid="baseButton-header"] {{ display: none !important; }}
section[data-testid="stSidebar"] > div:first-child > div:first-child > div:first-child > div:first-child > div:first-child > button {{ display: none !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    color: #6b6b60 !important; font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important; font-weight: 500 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 0.5rem 0.8rem !important; border-radius: 0 !important;
    transition: color 0.15s ease !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{ color: #f0ede6 !important; background: transparent !important; }}
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {{ background-color: #d4f542 !important; border-color: #d4f542 !important; border-radius: 0 !important; }}
section[data-testid="stSidebar"] button {{ font-size: 0 !important; color: transparent !important; }}
section[data-testid="stSidebar"] button * {{ font-size: 0 !important; color: transparent !important; }}
section[data-testid="stSidebar"] .material-icons, section[data-testid="stSidebar"] [class*="material-icons"], section[data-testid="stSidebar"] [class*="material-symbols"] {{ display: none !important; font-size: 0 !important; width: 0 !important; height: 0 !important; margin: 0 !important; visibility: hidden !important; }}
section[data-testid="stSidebar"] svg {{ display: none !important; }}
section[data-testid="stSidebar"] i {{ display: none !important; font-size: 0 !important; }}

.sb-header {{ padding: 0.8rem 0.5rem 1rem 0.5rem !important; margin-bottom: 0.8rem !important; border-bottom: 1px solid #2a2a26 !important; }}
.sb-header .title {{ font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.95rem !important; color: #f0ede6 !important; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 !important; }}
.sb-header .sub {{ font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important; color: #6b6b60 !important; margin: 0.15rem 0 0 0 !important; text-transform: uppercase; letter-spacing: 0.04em; }}

/* ---- headings ---- */
h1 {{ font-family: 'Syne', sans-serif !important; font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important; color: #f0ede6 !important; margin-bottom: 0.5rem !important; text-transform: uppercase; }}
h2 {{ font-family: 'Syne', sans-serif !important; font-size: 1.3rem !important; font-weight: 600 !important; letter-spacing: 0.02em !important; color: #f0ede6 !important; text-transform: uppercase; }}
h3 {{ font-family: 'Syne', sans-serif !important; font-size: 1rem !important; font-weight: 600 !important; color: #f0ede6 !important; text-transform: uppercase; letter-spacing: 0.04em; }}

/* ---- section header (Fraunces) ---- */
.sect-h {{ font-family: 'Fraunces', serif !important; margin: 2rem 0 0.8rem 0 !important; font-weight: 600 !important; font-size: 1.4rem !important; color: #f0ede6 !important; font-style: normal; }}
.sect-h .small {{ font-family: 'DM Mono', monospace !important; font-weight: 400 !important; font-size: 0.72rem !important; color: #6b6b60 !important; margin-left: 0.6rem !important; text-transform: uppercase; letter-spacing: 0.04em; }}
.sect-h .accent {{ font-family: 'Fraunces', serif !important; color: #d4f542 !important; font-style: italic; }}

/* ---- hero ---- */
.hero {{ background: #111110 !important; border: 1px solid #2a2a26 !important; padding: 2.2rem 2rem !important; margin-bottom: 1.5rem !important; }}
.hero h1 {{ font-family: 'Syne', sans-serif !important; color: #f0ede6 !important; font-size: 2.2rem !important; font-weight: 800 !important; text-transform: uppercase; letter-spacing: 0.02em; margin: 0 0 0.6rem 0 !important; }}
.hero h1 .accent {{ color: #d4f542 !important; }}
.hero p {{ font-family: 'DM Mono', monospace !important; color: #6b6b60 !important; font-size: 0.82rem !important; margin: 0 !important; max-width: 700px; line-height: 1.8 !important; }}
.hero-pill {{ display: inline-block !important; padding: 0.2rem 0.6rem !important; background: transparent !important; border: 1px solid #2a2a26 !important; font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important; color: #6b6b60 !important; margin-right: 0.3rem !important; margin-top: 0.6rem !important; text-transform: uppercase; letter-spacing: 0.04em; }}

/* ---- KPI metric cards ---- */
.kpi {{ background: #111110 !important; border: 1px solid #2a2a26 !important; border-left: 3px solid #d4f542 !important; padding: 1.1rem 1.2rem !important; height: 100%; }}
.kpi .label {{ font-family: 'Syne', sans-serif !important; color: #6b6b60 !important; font-size: 0.6rem !important; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; margin: 0 0 0.4rem 0 !important; }}
.kpi .value {{ font-family: 'Syne', sans-serif !important; color: #f0ede6 !important; font-size: 1.6rem !important; font-weight: 700 !important; margin: 0 !important; line-height: 1.1; letter-spacing: -0.02em; }}
.kpi .sub {{ font-family: 'DM Mono', monospace !important; color: #6b6b60 !important; font-size: 0.65rem !important; margin: 0.35rem 0 0 0 !important; }}

/* ---- prediction result card ---- */
.pred-card {{ background: #111110 !important; border: 1px solid #2a2a26 !important; border-left: 3px solid #d4f542 !important; padding: 1.5rem 1.8rem !important; margin-bottom: 1.2rem !important; }}
.pred-card .label {{ font-family: 'Syne', sans-serif !important; color: #6b6b60 !important; font-size: 0.65rem !important; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; margin: 0 !important; }}
.pred-card .value {{ font-family: 'Syne', sans-serif !important; color: #f0ede6 !important; font-size: 2.6rem !important; font-weight: 800 !important; margin: 0.2rem 0 0.4rem 0 !important; letter-spacing: -0.03em; }}
.pred-card .sub {{ font-family: 'DM Mono', monospace !important; color: #6b6b60 !important; font-size: 0.78rem !important; margin: 0 !important; }}

/* ---- footer ---- */
.app-footer {{ margin-top: 3rem !important; padding-top: 1rem !important; border-top: 1px solid #2a2a26 !important; color: #6b6b60 !important; font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important; text-align: center !important; text-transform: uppercase; letter-spacing: 0.04em; }}

/* ---- pills / badges ---- */
.pill {{ display: inline-block !important; padding: 0.15rem 0.5rem !important; background: transparent !important; color: #6b6b60 !important; border: 1px solid #2a2a26 !important; font-family: 'DM Mono', monospace !important; font-size: 0.62rem !important; font-weight: 500 !important; margin-right: 0.3rem !important; text-transform: uppercase; letter-spacing: 0.04em; }}
.pill-green {{ border-color: #d4f542 !important; color: #d4f542 !important; }}
.pill-red {{ border-color: #2a2a26 !important; color: #2a2a26 !important; }}
.pill-amber {{ border-color: #6b6b60 !important; color: #6b6b60 !important; }}

/* ---- forms ---- */
[data-testid="stForm"] {{ background: #111110 !important; padding: 1.5rem !important; border: 1px solid #2a2a26 !important; }}
[data-testid="stForm"] h3 {{ margin-top: 0 !important; }}

/* ---- selects / inputs ---- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
    background: #0a0a08 !important; border: 1px solid #2a2a26 !important; border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important; transition: border-color 0.15s ease !important;
}}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {{ border-color: #6b6b60 !important; }}
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {{ border-color: #d4f542 !important; box-shadow: none !important; }}
div[data-baseweb="select"] input, div[data-baseweb="input"] input {{
    color: #f0ede6 !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important;
}}
div[data-baseweb="select"] svg {{ fill: #6b6b60 !important; }}
div[data-baseweb="select"] ul {{ background: #111110 !important; border: 1px solid #2a2a26 !important; border-radius: 0 !important; }}

/* ---- slider ---- */
div[data-testid="stSlider"] [role="slider"] {{ background: #d4f542 !important; border: none !important; border-radius: 0 !important; }}
div[data-testid="stSlider"] div[data-testid="stTickBar"] {{ color: #6b6b60 !important; font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important; }}
div[data-testid="stSlider"] .st-bb {{ background: #2a2a26 !important; }}
div[data-testid="stSlider"] .st-ba {{ background: #d4f542 !important; }}

/* ---- number input ---- */
.stNumberInput div[data-baseweb="input"] {{ background: #0a0a08 !important; border: 1px solid #2a2a26 !important; border-radius: 0 !important; }}
.stNumberInput div[data-baseweb="input"]:focus-within {{ border-color: #d4f542 !important; }}
.stNumberInput button {{ background: #111110 !important; border: 1px solid #2a2a26 !important; color: #6b6b60 !important; border-radius: 0 !important; font-size: 0.7rem !important; }}
.stNumberInput button:hover {{ background: #d4f542 !important; color: #0a0a08 !important; }}

/* ---- buttons ---- */
.stButton button {{
    border-radius: 0 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em;
    padding: 0.5rem 1.4rem !important; transition: all 0.15s ease !important;
    border: 1px solid #2a2a26 !important; background: transparent !important; color: #6b6b60 !important;
}}
.stButton button:hover {{ border-color: #6b6b60 !important; color: #f0ede6 !important; }}
.stButton button[kind="primary"] {{ background: #d4f542 !important; color: #0a0a08 !important; border: none !important; }}
.stButton button[kind="primary"]:hover {{ background: #f0ede6 !important; color: #0a0a08 !important; }}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {{ gap: 0 !important; background: transparent !important; padding: 0 !important; border-bottom: 1px solid #2a2a26 !important; border-radius: 0 !important; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 0 !important; padding: 0.5rem 1rem !important;
    font-family: 'Syne', sans-serif !important; font-size: 0.72rem !important; font-weight: 500 !important;
    color: #6b6b60 !important; text-transform: uppercase; letter-spacing: 0.06em;
    transition: color 0.15s ease !important; border-bottom: 2px solid transparent !important;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: #f0ede6 !important; background: transparent !important; }}
.stTabs [aria-selected="true"] {{ background: transparent !important; color: #d4f542 !important; border-bottom: 2px solid #d4f542 !important; box-shadow: none !important; }}

/* ---- dataframe ---- */
div[data-testid="stDataFrame"] {{ border-radius: 0 !important; overflow: hidden !important; border: 1px solid #2a2a26 !important; background: #111110 !important; }}
div[data-testid="stDataFrame"] th {{ background: #0a0a08 !important; color: #6b6b60 !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 0.65rem !important; text-transform: uppercase; letter-spacing: 0.06em; padding: 0.6rem 0.8rem !important; }}
div[data-testid="stDataFrame"] td {{ color: #f0ede6 !important; font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important; padding: 0.5rem 0.8rem !important; border-top: 1px solid #2a2a26 !important; }}

/* ---- metric containers ---- */
div[data-testid="metric-container"] {{ background: #111110 !important; border: 1px solid #2a2a26 !important; border-left: 3px solid #d4f542 !important; padding: 1rem 1.2rem !important; }}
div[data-testid="metric-container"] label {{ font-family: 'Syne', sans-serif !important; color: #6b6b60 !important; font-size: 0.6rem !important; text-transform: uppercase; letter-spacing: 0.08em; }}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{ font-family: 'Syne', sans-serif !important; color: #f0ede6 !important; font-weight: 700 !important; font-size: 1.4rem !important; }}

/* ---- alerts ---- */
div[data-testid="stAlert"] {{ border-radius: 0 !important; border: 1px solid #2a2a26 !important; background: #111110 !important; }}
.stAlert > div:first-child {{ color: #f0ede6 !important; font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; }}

/* ---- expander ---- */
.streamlit-expanderHeader {{ background: #111110 !important; border-radius: 0 !important; color: #f0ede6 !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; border: 1px solid #2a2a26 !important; }}
.streamlit-expanderContent {{ border: 1px solid #2a2a26 !important; border-top: none !important; background: #111110 !important; }}

/* ---- progress ---- */
.stProgress div[role="progressbar"] > div {{ background: #d4f542 !important; border-radius: 0 !important; }}

/* ---- spinner ---- */
.stSpinner > div {{ border-color: #d4f542 !important; border-top-color: transparent !important; }}

/* ---- caption ---- */
.stCaption {{ color: #6b6b60 !important; font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.04em; }}

/* ---- file uploader ---- */
section[data-testid="stFileUploader"] {{ background: #111110 !important; border: 1px dashed #2a2a26 !important; padding: 1rem !important; }}

/* ---- download btn ---- */
.stDownloadButton button {{ background: transparent !important; border: 1px solid #2a2a26 !important; color: #6b6b60 !important; font-family: 'Syne', sans-serif !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.06em; border-radius: 0 !important; }}
.stDownloadButton button:hover {{ border-color: #d4f542 !important; color: #d4f542 !important; }}

/* ---- sidebar caption ---- */
section[data-testid="stSidebar"] .stCaption {{ color: #6b6b60 !important; }}

/* ---- markdown links ---- */
.stMarkdown a {{ color: #d4f542 !important; text-decoration: none !important; border-bottom: 1px solid transparent !important; }}
.stMarkdown a:hover {{ border-bottom-color: #d4f542 !important; }}

/* ---- scrollbar ---- */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: #0a0a08; }}
::-webkit-scrollbar-thumb {{ background: #2a2a26; }}
::-webkit-scrollbar-thumb:hover {{ background: #6b6b60; }}
::selection {{ background: #d4f542; color: #0a0a08; }}

/* ---- responsive ---- */
@media (max-width: 768px) {{
    .block-container {{ padding-top: 0.5rem !important; }}
    .hero {{ padding: 1.2rem !important; }}
    .hero h1 {{ font-size: 1.4rem !important; }}
    .pred-card .value {{ font-size: 1.8rem !important; }}
    .kpi .value {{ font-size: 1.2rem !important; }}
}}
</style>""", unsafe_allow_html=True)

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

MODEL_BLURB = {
    "Linear Regression": "Closed-form OLS baseline; fast, interpretable, but cannot capture non-linear interactions.",
    "Ridge Regression": "L2-regularised linear model; tames multicollinearity in the encoded categorical features.",
    "Lasso Regression": "L1-regularised linear model with implicit feature selection.",
    "Support Vector Regression": "RBF-kernel SVR; flexible non-linear baseline but sensitive to scaling and slow to train.",
    "Random Forest": "Bagged ensemble of decision trees; strong out-of-the-box performance, robust to outliers.",
    "Gradient Boosting": "Sequential tree boosting (sklearn); high accuracy at modest training cost.",
    "XGBoost": "Regularised gradient-boosted trees; the workhorse for tabular regression.",
    "AdaBoost": "Adaptive boosting of shallow trees; fast but more sensitive to label noise.",
    "MLP": "Two-layer fully-connected neural network with early stopping.",
    "Weighted Ensemble": "Top-3 performance-weighted voting of the strongest base learners (weights ∝ validation R²).",
}

DEFAULT_BRANDS = ["Toyota", "Honda", "BMW", "Mercedes", "Audi", "Ford", "Nissan", "Tesla"]


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_models():
    base_models: dict[str, object] = {}
    for name, path in MODEL_PATHS.items():
        if Path(path).exists():
            try:
                base_models[name] = joblib.load(path)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not load {name}: {exc}")

    # Reconstruct ensemble from base models + weights JSON
    # instead of loading from pickle to avoid serialization issues
    ensemble = None
    weights_path = MODELS_DIR / "ensemble_weights.json"
    if weights_path.exists() and base_models:
        try:
            from train_model import WeightedEnsemble
            weights_dict = json.loads(weights_path.read_text(encoding="utf-8"))
            # Create a dummy val_r2_scores dict from base_models names
            # (not needed for reconstruction, but WeightedEnsemble __init__ expects it)
            val_r2 = {name: 0.5 for name in weights_dict.keys()}
            # Create the ensemble with the base models
            ensemble = WeightedEnsemble(base_models, val_r2, top_k=None)
            # Override the weights with the saved weights (bypassing the filtering logic)
            ensemble.weights = {k: float(v) for k, v in weights_dict.items()}
            # Prune models to match weights
            ensemble.models = {k: base_models[k] for k in ensemble.weights.keys()}
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not reconstruct ensemble: {exc}")

    scaler = None
    if Path(SCALER_FILE).exists():
        scaler = joblib.load(SCALER_FILE)

    encoder = None
    if Path(ENCODER_FILE).exists():
        encoder = joblib.load(ENCODER_FILE)

    return base_models, ensemble, scaler, encoder


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict:
    out: dict = {}
    for fname in (
        "ensemble_weights.json",
        "all_metrics.json",
        "per_fold_r2.json",
        "wilcoxon_results.json",
    ):
        path = MODELS_DIR / fname
        out[fname] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    shap_path = MODELS_DIR / "shap_summary.png"
    out["shap_summary_png"] = shap_path if shap_path.exists() else None

    shap_csv = MODELS_DIR / "shap_importance.csv"
    out["shap_importance"] = pd.read_csv(shap_csv) if shap_csv.exists() else None

    return out


def _safe_options(encoder, col: str, fallback: list[str]) -> list[str]:
    if encoder and col in encoder:
        try:
            return sorted([str(x) for x in encoder[col].classes_])
        except Exception:  # noqa: BLE001
            return fallback
    return fallback


# Date-artefact model names produced by Excel mis-typing (e.g. "09-Mar"
# was originally a model code like "09/03" parsed as a date).
_DATE_MONTHS = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}


def _is_date_artefact(value: str) -> bool:
    parts = value.split("-")
    if len(parts) != 2:
        return False
    a, b = parts[0].strip(), parts[1].strip()
    return (a.isdigit() and b in _DATE_MONTHS) or (
        b.isdigit() and a in _DATE_MONTHS
    )


_ALL_CAPS_BRANDS = {"BMW", "GMC", "GAZ", "UAZ", "VAZ"}


def _pretty_brand(value: str) -> str:
    """Display name for brands. Encoder still receives the raw value."""
    s = str(value).strip()
    if s == "სხვა":
        return "Other"
    if s in _ALL_CAPS_BRANDS:
        return s
    return s.title()


def _pretty_model(value: str) -> str:
    s = str(value).strip()
    if _is_date_artefact(s):
        return f"{s} (legacy code)"
    return s


def _sort_brands(values: list[str]) -> list[str]:
    """Western/common brands first, Other / Georgian last, otherwise alpha."""
    preferred = ["BMW", "TOYOTA", "HONDA", "MERCEDES-BENZ", "AUDI",
                 "FORD", "NISSAN", "VOLKSWAGEN", "HYUNDAI", "CHEVROLET"]
    head = [v for v in preferred if v in values]
    tail = sorted(v for v in values if v not in head and v != "სხვა")
    other = ["სხვა"] if "სხვა" in values else []
    return head + tail + other


def _filter_models(values: list[str]) -> list[str]:
    """Hide the worst Excel date artefacts; keep them only at the very end."""
    clean = [v for v in values if not _is_date_artefact(v)]
    artefacts = [v for v in values if _is_date_artefact(v)]
    return sorted(clean) + sorted(artefacts)


def _persist_history(record: dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history: list = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            history = []
    history.append(record)
    HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")


def _read_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def _kpi(col, label: str, value: str, sub: str = "") -> None:
    sub_html = f'<p class="sub">{sub}</p>' if sub else ""
    html = (
        f'<div class="kpi"><p class="label">{label}</p>'
        f'<p class="value">{value}</p>{sub_html}</div>'
    )
    col.markdown(html, unsafe_allow_html=True)


def _section(title: str, sub: str = "", accent: str = "") -> None:
    extra = f'<span class="small">{sub}</span>' if sub else ""
    accent_html = f' <span class="accent">{accent}</span>' if accent else ""
    st.markdown(f'<div class="sect-h">{title}{accent_html}{extra}</div>', unsafe_allow_html=True)


def _predict_all(
    payload: dict,
    base_models: dict,
    ensemble,
    scaler,
    encoder,
) -> tuple[dict[str, float], float, str]:
    X = preprocess_input_data(payload, encoder, scaler)

    individual: dict[str, float] = {}
    for name, model in base_models.items():
        try:
            pred = float(np.asarray(model.predict(X)).reshape(-1)[0])
            individual[name] = max(0.0, pred)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{name} failed to predict: {exc}")

    if ensemble is not None:
        ensemble_pred = float(np.asarray(ensemble.predict(X)).reshape(-1)[0])
        label = "Weighted Ensemble"
    else:
        ensemble_pred = float(np.mean(list(individual.values()))) if individual else 0.0
        label = "Mean of Models"

    return individual, ensemble_pred, label


# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #

def show_home_page(metrics: dict | None, weights: dict | None, df: pd.DataFrame) -> None:
    ens = (metrics or {}).get("Weighted Ensemble", {})
    rf = (metrics or {}).get("Random Forest", {})
    xgb = (metrics or {}).get("XGBoost", {})

    st.markdown(
        f"""
        <div class="hero">
            <h1>Used-Car <span class="accent">Price</span> Predictor</h1>
            <p>A 9-model heterogeneous ensemble. Predict in seconds, then break the
            answer down model by model with full statistical rigour — 5-fold cross
            validation, Wilcoxon signed-rank test, SHAP feature importance.</p>
            <div>
                <span class="hero-pill">9 Base Models</span>
                <span class="hero-pill">Top-3 Weighted Voting</span>
                <span class="hero-pill">Real Kaggle Data · 14,747 rows</span>
                <span class="hero-pill">Wilcoxon @ α = 0.05</span>
                <span class="hero-pill">SHAP Explainability</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    _kpi(cols[0], "Ensemble Test R²",
         f"{ens.get('r2', float('nan')):.4f}" if ens else "—",
         "Held-out test set (2,824 rows)" if ens else "")
    _kpi(cols[1], "Ensemble RMSE",
         f"${ens.get('rmse', float('nan')):,.0f}" if ens else "—",
         f"MAE ${ens.get('mae', float('nan')):,.0f}" if ens else "")
    _kpi(cols[2], "Best Single Model",
         f"R² {rf.get('r2', float('nan')):.4f}" if rf else "—",
         "Random Forest" if rf else "")
    _kpi(cols[3], "Models Combined",
         f"{len(weights)}" if weights else "0",
         "Top-K = 3 (RF, XGB, GB)" if weights else "")

    st.markdown(" ")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        _section("How it works")
        st.markdown(
            """
            1. **Choose a car.** Brand, model, year, mileage, engine, fuel,
               transmission and condition.
            2. **The 9 base models predict.** Linear, Ridge, Lasso, SVR, Random
               Forest, Gradient Boosting, XGBoost, AdaBoost, and MLP each output
               their own price independently.
            3. **The top-3 ensemble combines them.** Models are ranked by
               held-out validation R², and the strongest three are blended with
               weights proportional to their R² (negative R² clipped to 0,
               weights normalised to 1).
            4. **You see everything.** The ensemble price, every individual
               prediction, the ensemble weights, and per-prediction SHAP values
               that explain *why* the model chose this number.
            """
        )

        _section("Methodology", "R²-weighted top-3 voting", "in one line")
        st.markdown(
            r"""
            Performance-weighted voting (Zhou, Wu & Tang, 2002) with selective
            top-K = 3 dilution control, where for each selected model
            $w_k \propto R^2_{k,\text{validation}}$ and
            $\sum_k w_k = 1$. Final prediction:
            $\hat y = \sum_{k} w_k \, \hat y_k$.
            """
        )

    with col_right:
        _section("Real Kaggle dataset")
        rows = len(df)
        st.markdown(
            f"""
            - **Source.** Kaggle *Car Price Prediction Challenge* (public mirror)
            - **After cleaning.** {rows:,} rows, 8 features
            - **Cleaning bounds.** Year 1990–2023, Price \\$1k–\\$66k, Mileage 150–298k mi, Z ≤ 3
            - **Train / test split.** 11,295 / 2,824 (80 / 20, stratified by Year decile)
            - **Median price.** \\${df['Price'].median():,.0f}
            - **Median mileage.** {int(df['Mileage'].median()):,} mi
            """
        )
        _section("Try it")
        st.markdown(
            "Open **Price Predictor** in the sidebar to make your first prediction. "
            "Already have a CSV? Use **Batch Predict** to score many cars at once."
        )

    st.markdown('<div class="app-footer">'
                f'Master\'s thesis project · {APP_VERSION} · '
                f'<a href="{GITHUB_URL}" target="_blank">GitHub</a>'
                '</div>', unsafe_allow_html=True)


@st.cache_data
def _brand_model_map(df: pd.DataFrame) -> dict[str, list[str]]:
    return df.groupby("Brand")["Model"].apply(lambda x: sorted(x.unique().tolist())).to_dict()


def show_predictor_page(base_models, ensemble, scaler, encoder, weights: dict | None,
                        artifacts: dict, df: pd.DataFrame) -> None:
    st.markdown("## Price Predictor")

    if not base_models or scaler is None or encoder is None:
        st.error("Models or preprocessors are missing. Run `python train_model.py` first.")
        return

    brand_options = _sort_brands(_safe_options(encoder, "Brand", DEFAULT_BRANDS))
    fuel_options = _safe_options(encoder, "Fuel Type", ["Petrol", "Diesel", "Hybrid", "CNG", "LPG", "Electric"])
    transmission_options = _safe_options(encoder, "Transmission", ["Manual", "Automatic", "Tiptronic", "Variator"])
    condition_options = _safe_options(encoder, "Condition", ["Sedan", "Hatchback", "Coupe"])
    all_model_options = _filter_models(_safe_options(encoder, "Model", ["Corolla"]))

    brand_model_map = _brand_model_map(df)

    def _get_models_for_brand(brand: str) -> list[str]:
        raw = brand_model_map.get(brand, [])
        valid = [m for m in all_model_options if m in raw]
        return _filter_models(valid) if valid else all_model_options[: min(len(all_model_options), 400)]

    st.markdown("### Car details")
    b1, b2, b3 = st.columns(3)
    with b1:
        brand = st.selectbox(
            "Brand", brand_options,
            index=brand_options.index("BMW") if "BMW" in brand_options else 0,
            format_func=_pretty_brand,
        )
    with b2:
        filtered = _get_models_for_brand(brand)
        model_value = st.selectbox(
            "Model", filtered, index=0,
            format_func=_pretty_model,
            help="Models filtered by selected brand.",
        )
    with b3:
        condition = st.selectbox(
            "Body type", condition_options,
            index=condition_options.index("Sedan") if "Sedan" in condition_options else 0,
            help="The vehicle's body shape (Sedan, Hatchback, …). The Kaggle column was named 'Condition'.",
        )

    with st.form("predictor_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            year = st.slider("Year", 1990, CURRENT_YEAR, 2018)
            engine_size = st.number_input("Engine size (L)", 0.5, 8.0, 2.0, 0.1)
            fuel_type = st.selectbox(
                "Fuel type", fuel_options,
                index=fuel_options.index("Petrol") if "Petrol" in fuel_options else 0,
            )
        with c2:
            mileage = st.number_input("Mileage (mi)", 0, 500_000, 60_000, 1000)
            transmission = st.selectbox(
                "Transmission", transmission_options,
                index=transmission_options.index("Automatic") if "Automatic" in transmission_options else 0,
            )
            st.markdown(" ")
            submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

    if not submitted:
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
        individual, ensemble_pred, label = _predict_all(payload, base_models, ensemble, scaler, encoder)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not run prediction: {exc}")
        return

    if not individual:
        st.error("No model produced a prediction.")
        return

    arr = np.fromiter(individual.values(), dtype=float)
    p_low = max(0.0, ensemble_pred - 1.96 * arr.std())
    p_high = ensemble_pred + 1.96 * arr.std()

    # Headline prediction card.
    st.markdown(
        f"""
        <div class="pred-card">
            <p class="label">{label} — predicted price</p>
            <p class="value">${ensemble_pred:,.0f}</p>
            <p class="sub">Inter-model 95% range: ${p_low:,.0f} — ${p_high:,.0f} ·
            {len(individual)} models contributed</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    _kpi(cols[0], "Min model", f"${arr.min():,.0f}",
         min(individual, key=individual.get))
    _kpi(cols[1], "Max model", f"${arr.max():,.0f}",
         max(individual, key=individual.get))
    _kpi(cols[2], "Std-dev",
         f"${arr.std():,.0f}",
         "Lower = stronger consensus")
    _kpi(cols[3], "Car age",
         f"{CURRENT_YEAR - year} years",
         f"Year {year}")

    pred_df = pd.DataFrame(
        {
            "Model": list(individual.keys()),
            "Prediction": list(individual.values()),
            "Weight": [(weights or {}).get(name, 0.0) for name in individual.keys()],
        }
    ).sort_values("Prediction")

    _section("Predictions per model", "colour = ensemble weight; vertical line = ensemble price", "per model")
    fig = px.bar(
        pred_df, x="Prediction", y="Model", color="Weight", orientation="h",
        color_continuous_scale="Viridis",
        height=380,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(title="Weight"),
        xaxis_title="Predicted price ($)",
        yaxis_title=None,
    )
    fig.add_vline(x=ensemble_pred, line_dash="dash", line_color=ACCENT,
                  annotation_text=f"Ensemble = ${ensemble_pred:,.0f}",
                  annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    if weights:
        _section("Top-3 ensemble weights", "weights are proportional to validation R²")
        wdf = pd.DataFrame(
            {"Model": list(weights.keys()), "Weight": list(weights.values())}
        ).sort_values("Weight", ascending=False)
        wdf["Weight (%)"] = wdf["Weight"] * 100
        st.dataframe(
            wdf[["Model", "Weight", "Weight (%)"]].style.format(
                {"Weight": "{:.4f}", "Weight (%)": "{:.2f}%"}
            ),
            use_container_width=False, hide_index=True,
        )

    # Per-prediction SHAP if XGBoost is loaded.
    if "XGBoost" in base_models:
        try:
            import shap
            xgb_model = base_models["XGBoost"]
            X = preprocess_input_data(payload, encoder, scaler)
            explainer = shap.TreeExplainer(xgb_model)
            sv = explainer.shap_values(X)
            base = float(explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray)
                         else explainer.expected_value[0])
            # Order must match preprocess_input_data:
            # NUMERICAL_FEATURES + CATEGORICAL_FEATURES.
            feature_names = ["Year", "Car Age", "Engine Size", "Mileage",
                             "Brand", "Fuel Type", "Transmission",
                             "Body Type", "Model"]
            sv_arr = np.asarray(sv).reshape(-1)[: len(feature_names)]
            # Pad / truncate as defensive guard.
            if len(sv_arr) < len(feature_names):
                feature_names = feature_names[: len(sv_arr)]
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP value (USD)": sv_arr,
                "Effect": np.where(sv_arr > 0, "↑ raises price", "↓ lowers price"),
            }).sort_values("SHAP value (USD)", key=lambda s: s.abs(), ascending=True)

            _section("Why this prediction (SHAP)",
                     f"baseline = ${base:,.0f}; bars push the prediction up or down")
            fig = px.bar(
                shap_df, x="SHAP value (USD)", y="Feature", orientation="h",
                color="SHAP value (USD)", color_continuous_scale="RdBu",
                color_continuous_midpoint=0.0,
                height=380,
            )
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Per-prediction SHAP unavailable: {exc}")

    # Persist + session.
    history_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": payload,
        "predictions": individual,
        "ensemble": ensemble_pred,
    }
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    st.session_state.prediction_history.append(history_payload)
    try:
        _persist_history(history_payload)
    except Exception:  # noqa: BLE001
        pass


def show_compare_page(base_models, ensemble, scaler, encoder, weights: dict | None) -> None:
    st.markdown("## Compare two cars")
    st.caption("Score two cars side by side and see which is the better buy.")

    if not base_models or scaler is None or encoder is None:
        st.error("Models or preprocessors are missing. Run `python train_model.py` first.")
        return

    brands = _sort_brands(_safe_options(encoder, "Brand", DEFAULT_BRANDS))
    fuels = _safe_options(encoder, "Fuel Type", ["Petrol", "Diesel", "Hybrid"])
    transmissions = _safe_options(encoder, "Transmission", ["Manual", "Automatic"])
    conditions = _safe_options(encoder, "Condition", ["Sedan", "Hatchback", "Coupe"])
    models = _filter_models(_safe_options(encoder, "Model", ["Corolla"]))

    def _car_form(side: str, defaults: dict) -> dict:
        st.markdown(f"### Car {side}")
        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox(f"Brand ({side})", brands,
                                 index=brands.index(defaults["Brand"]) if defaults["Brand"] in brands else 0,
                                 format_func=_pretty_brand)
            year = st.slider(f"Year ({side})", 1990, CURRENT_YEAR, defaults["Year"])
            engine = st.number_input(f"Engine (L) ({side})", 0.5, 8.0, defaults["Engine Size"], 0.1)
            fuel = st.selectbox(f"Fuel ({side})", fuels,
                                index=fuels.index(defaults["Fuel Type"]) if defaults["Fuel Type"] in fuels else 0)
        with c2:
            model_v = st.selectbox(f"Model ({side})", models[:400], index=0,
                                   format_func=_pretty_model)
            mileage = st.number_input(f"Mileage (mi) ({side})", 0, 500_000, defaults["Mileage"], 1000)
            tx = st.selectbox(f"Transmission ({side})", transmissions,
                              index=transmissions.index(defaults["Transmission"]) if defaults["Transmission"] in transmissions else 0)
            cond = st.selectbox(f"Body type ({side})", conditions,
                                index=conditions.index(defaults["Condition"]) if defaults["Condition"] in conditions else 0)
        return {
            "Brand": brand, "Model": model_v, "Year": year, "Engine Size": engine,
            "Mileage": mileage, "Fuel Type": fuel, "Transmission": tx, "Condition": cond,
        }

    petrol = "Petrol" if "Petrol" in fuels else fuels[0]
    auto = "Automatic" if "Automatic" in transmissions else transmissions[0]
    sedan = "Sedan" if "Sedan" in conditions else conditions[0]
    bmw = "BMW" if "BMW" in brands else brands[0]
    toyota = "TOYOTA" if "TOYOTA" in brands else (brands[1] if len(brands) > 1 else brands[0])
    col_a, col_b = st.columns(2)
    with col_a:
        car_a = _car_form("A", {"Brand": bmw, "Year": 2018, "Engine Size": 2.0,
                                 "Fuel Type": petrol, "Mileage": 60000,
                                 "Transmission": auto, "Condition": sedan})
    with col_b:
        car_b = _car_form("B", {"Brand": toyota, "Year": 2015, "Engine Size": 1.8,
                                 "Fuel Type": petrol, "Mileage": 90000,
                                 "Transmission": auto, "Condition": sedan})

    if not st.button("Compare", type="primary", use_container_width=True):
        return

    pa, ea, _ = _predict_all(car_a, base_models, ensemble, scaler, encoder)
    pb, eb, _ = _predict_all(car_b, base_models, ensemble, scaler, encoder)

    cols = st.columns(2)
    _kpi(cols[0], "Car A — Ensemble", f"${ea:,.0f}",
         f"{_pretty_brand(car_a['Brand'])} · {car_a['Year']} · {int(car_a['Mileage']):,} mi")
    _kpi(cols[1], "Car B — Ensemble", f"${eb:,.0f}",
         f"{_pretty_brand(car_b['Brand'])} · {car_b['Year']} · {int(car_b['Mileage']):,} mi")

    diff = ea - eb
    if abs(diff) < 1:
        st.info("The two cars price almost identically.")
    else:
        which = "A" if diff > 0 else "B"
        st.markdown(
            f"<span class='pill'>Car {which} prices ${abs(diff):,.0f} higher than the other</span>",
            unsafe_allow_html=True,
        )

    # Side-by-side per-model bar chart.
    rows = []
    for name in pa.keys():
        rows.append({"Model": name, "Car": "A", "Prediction": pa[name]})
        rows.append({"Model": name, "Car": "B", "Prediction": pb.get(name, np.nan)})
    rows.append({"Model": "Ensemble", "Car": "A", "Prediction": ea})
    rows.append({"Model": "Ensemble", "Car": "B", "Prediction": eb})
    cmp_df = pd.DataFrame(rows)
    fig = px.bar(cmp_df, x="Prediction", y="Model", color="Car", orientation="h", barmode="group", height=420)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_title="Predicted price ($)", yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def show_batch_page(base_models, ensemble, scaler, encoder) -> None:
    st.markdown("## Batch Predict")
    st.caption("Upload a CSV (columns: Brand, Model, Year, Engine Size, Mileage, "
               "Fuel Type, Transmission, Condition) and get predictions for every row.")

    if not base_models or scaler is None or encoder is None:
        st.error("Models or preprocessors are missing. Run `python train_model.py` first.")
        return

    sample = pd.DataFrame([{
        "Brand": "BMW", "Model": "X5", "Year": 2018, "Engine Size": 3.0,
        "Mileage": 60000, "Fuel Type": "Petrol",
        "Transmission": "Automatic", "Condition": "Sedan",
    }, {
        "Brand": "TOYOTA", "Model": "Corolla", "Year": 2015, "Engine Size": 1.6,
        "Mileage": 95000, "Fuel Type": "Petrol",
        "Transmission": "Manual", "Condition": "Hatchback",
    }])
    csv_bytes = sample.to_csv(index=False).encode("utf-8")
    st.download_button("Download a sample CSV template", csv_bytes,
                       "car_input_sample.csv", "text/csv")

    f = st.file_uploader("Drop your CSV here", type=["csv"])
    if f is None:
        return

    try:
        df_in = pd.read_csv(f)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not parse CSV: {exc}")
        return

    required = {"Brand", "Model", "Year", "Engine Size", "Mileage",
                "Fuel Type", "Transmission", "Condition"}
    missing = required - set(df_in.columns)
    if missing:
        st.error(f"CSV missing columns: {sorted(missing)}")
        return

    progress = st.progress(0.0, text="Running predictions…")
    out_rows: list[dict] = []
    for i, row in enumerate(df_in.to_dict(orient="records")):
        try:
            individual, ens_pred, _ = _predict_all(row, base_models, ensemble, scaler, encoder)
        except Exception:  # noqa: BLE001
            individual, ens_pred = {}, float("nan")
        out_rows.append({**row,
                         **{f"{k} ($)": v for k, v in individual.items()},
                         "Ensemble Price ($)": ens_pred})
        progress.progress((i + 1) / max(1, len(df_in)),
                          text=f"Predicted {i + 1} / {len(df_in)}")
    progress.empty()

    out_df = pd.DataFrame(out_rows)
    st.success(f"Done — {len(out_df)} rows scored.")
    st.dataframe(out_df.head(20), use_container_width=True, hide_index=True)
    csv_out = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full predictions as CSV", csv_out,
                       "car_predictions.csv", "text/csv", use_container_width=True)


def show_eda_page(df: pd.DataFrame) -> None:
    st.markdown("## Data Analysis")

    cols = st.columns(4)
    _kpi(cols[0], "Rows", f"{len(df):,}")
    _kpi(cols[1], "Brands", f"{df['Brand'].nunique():,}",
         f"Top: {df['Brand'].value_counts().idxmax()}")
    _kpi(cols[2], "Median price", f"${df['Price'].median():,.0f}",
         f"Mean ${df['Price'].mean():,.0f}")
    _kpi(cols[3], "Median mileage", f"{int(df['Mileage'].median()):,} mi",
         f"Mean {int(df['Mileage'].mean()):,}")

    tabs = st.tabs(["Distributions", "Year & Mileage", "Brand mix", "Correlations", "Sample rows"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Price", nbins=60, title="Price distribution",
                               color_discrete_sequence=[ACCENT])
            fig.add_vline(x=df["Price"].median(), line_dash="dash",
                          annotation_text="median")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="Mileage", nbins=60, title="Mileage distribution",
                               color_discrete_sequence=[ACCENT])
            fig.add_vline(x=df["Mileage"].median(), line_dash="dash",
                          annotation_text="median")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                df.sample(min(2500, len(df)), random_state=42),
                x="Year", y="Price", opacity=0.5,
                color_discrete_sequence=[ACCENT],
                title="Price vs Year",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(
                df.sample(min(2500, len(df)), random_state=42),
                x="Mileage", y="Price", opacity=0.5,
                color_discrete_sequence=[ACCENT],
                title="Price vs Mileage",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        depreciation = (df.groupby("Year")["Price"].median().reset_index()
                          .sort_values("Year"))
        fig = px.line(depreciation, x="Year", y="Price", markers=True,
                      title="Median price by Year (depreciation curve)",
                      color_discrete_sequence=[ACCENT])
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                          yaxis_title="Median price ($)")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        brand_stats = (df.groupby("Brand")
                         .agg(count=("Price", "size"),
                              mean_price=("Price", "mean"),
                              median_price=("Price", "median"))
                         .reset_index())
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(brand_stats.sort_values("count", ascending=False).head(15),
                         x="count", y="Brand", orientation="h",
                         title="Listings per brand (top 15)",
                         color_discrete_sequence=[ACCENT])
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(brand_stats.sort_values("median_price", ascending=False).head(15),
                         x="median_price", y="Brand", orientation="h",
                         title="Median price per brand (top 15)",
                         color_discrete_sequence=[ACCENT])
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            brand_stats.sort_values("count", ascending=False).style.format(
                {"mean_price": "${:,.0f}", "median_price": "${:,.0f}"}
            ),
            use_container_width=True, hide_index=True,
        )

    with tabs[3]:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu", zmin=-1, zmax=1,
            title="Correlation heatmap (numeric features)",
            text_auto=".2f",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)


def show_model_details_page(metrics: dict | None, weights: dict | None,
                            per_fold: dict | None,
                            shap_png: Path | None,
                            shap_imp: pd.DataFrame | None) -> None:
    st.markdown("## Model Details")
    if not metrics:
        st.info("No metrics yet. Run `python train_model.py` first.")
        return

    ens = metrics.get("Weighted Ensemble", {})
    cols = st.columns(4)
    _kpi(cols[0], "Ensemble R²", f"{ens.get('r2', float('nan')):.4f}", "Test set")
    _kpi(cols[1], "Ensemble RMSE", f"${ens.get('rmse', float('nan')):,.0f}")
    _kpi(cols[2], "Ensemble MAE", f"${ens.get('mae', float('nan')):,.0f}")
    _kpi(cols[3], "MAPE", f"{ens.get('mape', float('nan')):.1f}%",
         "Inflated by low-priced cars")

    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "R²": m.get("r2", float("nan")),
            "RMSE": m.get("rmse", float("nan")),
            "MAE": m.get("mae", float("nan")),
            "MAPE %": m.get("mape", float("nan")),
        })
    df_m = pd.DataFrame(rows).sort_values("R²", ascending=False)

    _section("Held-out test metrics — all models")
    st.dataframe(
        df_m.style.format({"R²": "{:.4f}", "RMSE": "${:,.0f}",
                           "MAE": "${:,.0f}", "MAPE %": "{:.1f}"})
              .background_gradient(subset=["R²"], cmap="Blues")
              .background_gradient(subset=["RMSE", "MAE"], cmap="Reds_r"),
        use_container_width=True, hide_index=True,
    )

    fig = px.bar(df_m, x="Model", y="R²",
                 color="R²", color_continuous_scale="Blues",
                 title="Test R² per model",
                 height=380)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                      coloraxis_showscale=False, xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    if per_fold:
        cv_df = pd.DataFrame(per_fold)
        cv_summary = pd.DataFrame({
            "Model": cv_df.columns,
            "CV R² (mean)": cv_df.mean().values,
            "CV R² (std)": cv_df.std().values,
        })
        # Test R² per model
        cv_summary["Test R²"] = cv_summary["Model"].map(
            lambda m: metrics.get(m, {}).get("r2", float("nan"))
        )
        cv_summary["Δ overfit"] = cv_summary["Test R²"] - cv_summary["CV R² (mean)"]

        _section("CV vs Test R²", "small Δ = consistent generalisation")
        st.dataframe(
            cv_summary.sort_values("CV R² (mean)", ascending=False).style.format({
                "CV R² (mean)": "{:.4f}", "CV R² (std)": "±{:.4f}",
                "Test R²": "{:.4f}", "Δ overfit": "{:+.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

        # Compare bar
        plot_df = cv_summary.melt(id_vars=["Model"],
                                  value_vars=["CV R² (mean)", "Test R²"],
                                  var_name="Source", value_name="R²")
        fig = px.bar(plot_df, x="Model", y="R²", color="Source",
                     barmode="group", title="CV mean R² vs Test R²", height=380)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    if weights:
        _section("Top-3 ensemble weights")
        wdf = pd.DataFrame(
            {"Model": list(weights.keys()), "Weight": list(weights.values())}
        ).sort_values("Weight", ascending=False)
        fig = px.bar(wdf, x="Weight", y="Model", orientation="h",
                     color="Weight", color_continuous_scale="Viridis",
                     height=260)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    _section("Model card — what each does")
    for name, blurb in MODEL_BLURB.items():
        if name in metrics:
            r2 = metrics[name].get("r2", float("nan"))
            st.markdown(
                f"**{name}** &nbsp;<span class='pill'>R² {r2:.4f}</span>"
                f"<br><span style='color:#475569;'>{blurb}</span>",
                unsafe_allow_html=True,
            )

    if shap_imp is not None:
        _section("SHAP feature importance (XGBoost, mean |SHAP|)")
        shap_top = shap_imp.head(10)
        fig = px.bar(shap_top, x="mean_abs_shap", y="feature",
                     orientation="h", color="mean_abs_shap",
                     color_continuous_scale="Plasma", height=380)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          coloraxis_showscale=False,
                          xaxis_title="Mean |SHAP| (USD)", yaxis_title=None,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    if shap_png is not None:
        st.image(str(shap_png), caption="SHAP summary plot — XGBoost")


def show_statistical_tests_page(per_fold: dict | None, wilcoxon: list | None) -> None:
    st.markdown("## Statistical Tests")

    if per_fold:
        df_pf = pd.DataFrame(per_fold)
        # Long format for boxplot
        long_df = df_pf.melt(var_name="Model", value_name="R²")
        # Sort by mean R²
        order = df_pf.mean().sort_values(ascending=False).index.tolist()
        long_df["Model"] = pd.Categorical(long_df["Model"], categories=order, ordered=True)
        long_df = long_df.sort_values("Model")

        _section("5-fold CV R² distribution",
                 "boxplot per model; tighter = more stable")
        fig = px.box(long_df, x="Model", y="R²", points="all",
                     color="Model",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10),
                          xaxis_tickangle=-25, showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

        _section("Per-fold R² table")
        df_show = df_pf.copy()
        df_show.index = [f"Fold {i + 1}" for i in range(len(df_show))]
        st.dataframe(df_show.style.format("{:.4f}"), use_container_width=True)

    if wilcoxon:
        _section("Wilcoxon signed-rank test",
                 "ensemble vs each base model; α = 0.05 (one-sided, ensemble > base)")
        wdf = pd.DataFrame(wilcoxon)
        wdf = wdf[["Comparison", "Ensemble mean R²", "Base mean R²", "Mean Δ", "W",
                    "p-value (one-sided)", "Significant @ α=0.05"]].copy()

        def _highlight(row):
            color = "background-color: #dcfce7;" if row["Significant @ α=0.05"] else ""
            return [color] * len(row)

        st.dataframe(
            wdf.style
              .apply(_highlight, axis=1)
              .format({
                  "Ensemble mean R²": "{:.4f}",
                  "Base mean R²": "{:.4f}",
                  "Mean Δ": "{:+.4f}",
                  "W": "{:.1f}",
                  "p-value (one-sided)": "{:.4f}",
              }),
            use_container_width=True, hide_index=True,
        )

        wins = sum(1 for r in wilcoxon if r.get("Significant @ α=0.05"))
        st.markdown(
            f"<span class='pill pill-green'>Significant wins: {wins} / {len(wilcoxon)}</span> "
            "&nbsp; The ensemble's improvement is statistically significant against "
            f"{wins} of {len(wilcoxon)} base models at the 5% level.",
            unsafe_allow_html=True,
        )
    else:
        st.info("No Wilcoxon results yet. Run `python wilcoxon_test.py` after training.")


def show_data_quality_page(df: pd.DataFrame) -> None:
    st.markdown("## Data Quality")
    report = generate_quality_report(df)

    qs = report.get("Quality Score", 0.0)
    cols = st.columns(4)
    _kpi(cols[0], "Quality score", f"{qs:.1f} / 100",
         "Completeness − duplicates%")
    overview = report.get("Dataset Overview", {})
    _kpi(cols[1], "Records", f"{overview.get('Total Records', 0):,}")
    _kpi(cols[2], "Features", f"{overview.get('Total Features', 0)}")
    _kpi(cols[3], "Memory", f"{overview.get('Memory Usage (MB)', 0):.1f} MB")

    qm = report.get("Data Quality Metrics", {})
    cols = st.columns(3)
    _kpi(cols[0], "Completeness", f"{qm.get('Completeness %', 0):.2f}%")
    _kpi(cols[1], "Duplicates", f"{qm.get('Duplicate Rows', 0):,}")
    _kpi(cols[2], "Total nulls", f"{qm.get('Null Values', 0):,}")

    feature_rows = []
    for col, info in report.get("Feature Analysis", {}).items():
        feature_rows.append({
            "Feature": col,
            "Type": info.get("Data Type", ""),
            "Non-null": info.get("Non-Null Count", 0),
            "Nulls": info.get("Null Count", 0),
            "Unique": info.get("Unique Values", 0),
        })
    feat_df = pd.DataFrame(feature_rows)
    _section("Feature audit")
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    _section("Outlier inspection")
    target_cols = [c for c in ["Price", "Mileage", "Engine Size"] if c in df.columns]
    if target_cols:
        cs = st.columns(len(target_cols))
        for c, col_name in zip(cs, target_cols):
            with c:
                fig = px.box(df, y=col_name, color_discrete_sequence=[ACCENT])
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
                st.plotly_chart(fig, use_container_width=True)


def show_history_page() -> None:
    st.markdown("## Prediction History")
    persisted = _read_history()
    session = st.session_state.get("prediction_history", [])
    history = persisted or session
    if not history:
        st.info("No predictions yet. Make one from the Price Predictor page.")
        return
    rows = []
    for h in history[-200:][::-1]:  # newest first, last 200
        rows.append({
            "Time": h.get("timestamp", ""),
            "Brand": h["input"].get("Brand", ""),
            "Model": h["input"].get("Model", ""),
            "Year": h["input"].get("Year", ""),
            "Mileage": h["input"].get("Mileage", ""),
            "Ensemble price": h.get("ensemble", float("nan")),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"Ensemble price": "${:,.0f}"}),
                 use_container_width=True, hide_index=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download history as CSV", csv_bytes,
                       "prediction_history.csv", "text/csv")


def show_about_page(metrics: dict | None) -> None:
    st.markdown("## About this project")
    st.markdown(
        f"""
        This dashboard accompanies a Master's thesis on **heterogeneous
        regression ensembles for used-car price prediction**. The full
        methodology and results are documented in the thesis report; the code
        is open-source on GitHub.

        **Repository.** [{GITHUB_URL}]({GITHUB_URL})

        ### Methodology in three lines
        1. Train **9 heterogeneous base regressors** (Linear, Ridge, Lasso,
           SVR, Random Forest, Gradient Boosting, XGBoost, AdaBoost, MLP) with
           shared preprocessing and `RandomizedSearchCV` tuning.
        2. Combine via a **performance-weighted top-3 ensemble** with
           $w_k \\propto R^2_{{k,\\text{{validation}}}}$; weights normalise to 1.
        3. Validate with **5-fold CV** and a **Wilcoxon signed-rank test**;
           explain with **SHAP** TreeExplainer.

        ### Data
        - **Source.** Kaggle *Car Price Prediction Challenge* (public mirror).
        - **Cleaning.** 19,237 → 14,747 rows after Z-score outlier removal,
          missing-value imputation, and Car_Age engineering.

        ### Headline result
        """
    )
    if metrics and "Weighted Ensemble" in metrics:
        ens = metrics["Weighted Ensemble"]
        st.markdown(
            f"- Top-3 weighted ensemble: **R² = {ens['r2']:.4f}**, "
            f"RMSE **\\${ens['rmse']:,.0f}**, MAE **\\${ens['mae']:,.0f}** on a "
            f"held-out test set of 2,824 rows."
        )

    st.markdown(
        """
        ### Cite as
        ```
        Abbassi, J. (2025). Heterogeneous regression ensemble for used-car
        price prediction. Master's thesis. https://github.com/jasserabbassi/car-price-prediction
        ```
        """
    )


# --------------------------------------------------------------------------- #
# Sidebar + main                                                               #
# --------------------------------------------------------------------------- #

PAGES = [
    "Home",
    "Price Predictor",
    "Compare Cars",
    "Batch Predict",
    "Data Analysis",
    "Model Details",
    "Statistical Tests",
    "Data Quality",
    "Prediction History",
    "About",
]


def sidebar_header() -> None:
    st.sidebar.markdown(
        f"""
        <div class="sb-header">
            <p class="title">Car Price Predictor</p>
            <p class="sub">9-model weighted ensemble · {APP_VERSION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer(metrics: dict | None) -> None:
    st.sidebar.markdown("---")
    if metrics and "Weighted Ensemble" in metrics:
        ens = metrics["Weighted Ensemble"]
        st.sidebar.caption(
            f"**Live model:** {len([m for m in metrics if m != 'Weighted Ensemble'])} base + 1 ensemble  \n"
            f"**Test R²:** {ens['r2']:.4f}  \n"
            f"**Test RMSE:** ${ens['rmse']:,.0f}"
        )
    st.sidebar.markdown(
        f"[GitHub repo]({GITHUB_URL})  \n"
        "[Open issues]({})".format(f"{GITHUB_URL}/issues")
    )


def main() -> None:
    _inject_css()
    sidebar_header()
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

    base_models, ensemble, scaler, encoder = load_models()
    df = load_dataset()
    artifacts = load_artifacts()
    weights = artifacts["ensemble_weights.json"]
    metrics = artifacts["all_metrics.json"]

    sidebar_footer(metrics)

    if page == "Home":
        show_home_page(metrics, weights, df)
    elif page == "Price Predictor":
        show_predictor_page(base_models, ensemble, scaler, encoder, weights, artifacts, df)
    elif page == "Compare Cars":
        show_compare_page(base_models, ensemble, scaler, encoder, weights)
    elif page == "Batch Predict":
        show_batch_page(base_models, ensemble, scaler, encoder)
    elif page == "Data Analysis":
        show_eda_page(df)
    elif page == "Model Details":
        show_model_details_page(
            metrics, weights, artifacts["per_fold_r2.json"],
            artifacts["shap_summary_png"], artifacts["shap_importance"]
        )
    elif page == "Statistical Tests":
        show_statistical_tests_page(
            artifacts["per_fold_r2.json"], artifacts["wilcoxon_results.json"]
        )
    elif page == "Data Quality":
        show_data_quality_page(df)
    elif page == "Prediction History":
        show_history_page()
    elif page == "About":
        show_about_page(metrics)


if __name__ == "__main__":
    main()
