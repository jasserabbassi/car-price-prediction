"""FastAPI Model Server — Production API for Car Price Prediction.

Run with::

    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /                  service welcome
    GET  /health            liveness check
    GET  /model-info        model metadata
    GET  /ensemble-weights  weighted-ensemble weights
    POST /predict           single-row prediction
    POST /batch-predict     multi-row prediction
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    CATEGORICAL_FEATURES,
    CURRENT_YEAR,
    ENCODER_FILE,
    MODEL_ENSEMBLE,
    MODEL_LINEAR,
    MODEL_RIDGE,
    MODEL_LASSO,
    MODEL_SVR,
    MODEL_RANDOM_FOREST,
    MODEL_GRADIENT_BOOSTING,
    MODEL_XGBOOST,
    MODEL_ADABOOST,
    MODEL_MLP,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    SCALER_FILE,
)

# Importing train_model registers WeightedEnsemble on the running ``__main__``
# (uvicorn here), so pickle.load can resolve ensembles serialised by older
# train_model.py runs that ran as ``__main__`` themselves.
from train_model import WeightedEnsemble  # noqa: F401

# --------------------------------------------------------------------------- #
# Model loading                                                                #
# --------------------------------------------------------------------------- #

MODEL_PATHS: Dict[str, Path] = {
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


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_models() -> Dict[str, object]:
    loaded: Dict[str, object] = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            try:
                loaded[name] = _load_pickle(path)
            except Exception as exc:  # noqa: BLE001 - warn-not-fail at startup
                print(f"[api] WARNING: failed to load {name} from {path}: {exc}")
        else:
            print(f"[api] WARNING: {path} not found, skipping {name}")
    return loaded


def _load_optional_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


print("[api] loading models …")
BASE_MODELS = _load_models()
ENSEMBLE = _load_pickle(MODEL_ENSEMBLE) if MODEL_ENSEMBLE.exists() else None
SCALER = _load_pickle(SCALER_FILE) if SCALER_FILE.exists() else None
ENCODER = _load_pickle(ENCODER_FILE) if ENCODER_FILE.exists() else None
ENSEMBLE_WEIGHTS = _load_optional_json(MODELS_DIR / "ensemble_weights.json")
ALL_METRICS = _load_optional_json(MODELS_DIR / "all_metrics.json")

print(f"[api] loaded {len(BASE_MODELS)} base models. Ensemble: {ENSEMBLE is not None}.")

# --------------------------------------------------------------------------- #
# FastAPI app                                                                  #
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Car Price Prediction API",
    description="9-model ensemble ML API with performance-weighted voting.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Pydantic schemas                                                             #
# --------------------------------------------------------------------------- #


class CarData(BaseModel):
    """Input payload for a single car."""

    Brand: str
    Model: str
    Year: int = Field(ge=1970, le=2030)
    Engine_Size: float = Field(ge=0.4, le=10.0, alias="Engine Size")
    Mileage: int = Field(ge=0)
    Fuel_Type: str = Field(alias="Fuel Type")
    Transmission: str
    Condition: str

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    individual_predictions: Dict[str, float]
    ensemble_weighted: float
    ensemble_weights: Dict[str, float]
    confidence_range: Dict[str, float]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #


def _prepare_input(payload: dict) -> np.ndarray:
    """Mirror utils.preprocess_input_data() but inline so the API has no Streamlit-only dep."""
    if SCALER is None or ENCODER is None:
        raise HTTPException(status_code=503, detail="Scaler/encoder not loaded — train models first.")

    row = {
        "Brand": payload.get("Brand"),
        "Model": payload.get("Model"),
        "Year": payload.get("Year"),
        "Engine Size": payload.get("Engine_Size", payload.get("Engine Size")),
        "Mileage": payload.get("Mileage"),
        "Fuel Type": payload.get("Fuel_Type", payload.get("Fuel Type")),
        "Transmission": payload.get("Transmission"),
        "Condition": payload.get("Condition"),
    }
    row["Car_Age"] = float(CURRENT_YEAR - int(row["Year"]))

    df = pd.DataFrame([row])

    for col in CATEGORICAL_FEATURES:
        if col in ENCODER:
            value = str(df[col].iloc[0])
            valid = list(ENCODER[col].classes_)
            if value not in valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown {col!r} value {value!r}. Valid options: {valid[:25]}",
                )
            df[col] = ENCODER[col].transform(df[col].astype(str))

    feature_order = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_order].astype(float).values
    return SCALER.transform(X)


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #


@app.get("/")
async def root() -> dict:
    return {
        "service": "Car Price Prediction API",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/model-info",
            "/ensemble-weights",
            "/predict",
            "/batch-predict",
        ],
    }


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "base_models_loaded": list(BASE_MODELS.keys()),
        "ensemble_loaded": ENSEMBLE is not None,
        "scaler_loaded": SCALER is not None,
        "encoder_loaded": ENCODER is not None,
    }


@app.get("/model-info")
async def model_info() -> dict:
    info = {}
    for name, model in BASE_MODELS.items():
        info[name] = {
            "type": type(model).__name__,
            "test_metrics": ALL_METRICS.get(name, {}),
        }
    info["Weighted Ensemble"] = {
        "type": "WeightedEnsemble",
        "test_metrics": ALL_METRICS.get("Weighted Ensemble", {}),
        "n_components": len(ENSEMBLE_WEIGHTS) if ENSEMBLE_WEIGHTS else 0,
    }
    return info


@app.get("/ensemble-weights")
async def ensemble_weights() -> dict:
    if not ENSEMBLE_WEIGHTS:
        raise HTTPException(status_code=503, detail="Ensemble weights not available.")
    return ENSEMBLE_WEIGHTS


@app.post("/predict", response_model=PredictionResponse)
async def predict(car_data: CarData) -> PredictionResponse:
    if not BASE_MODELS:
        raise HTTPException(status_code=503, detail="No models loaded.")
    X = _prepare_input(car_data.dict(by_alias=True))

    individual: Dict[str, float] = {}
    for name, model in BASE_MODELS.items():
        try:
            pred = float(np.asarray(model.predict(X)).reshape(-1)[0])
            individual[name] = max(0.0, pred)
        except Exception as exc:  # noqa: BLE001
            print(f"[predict] {name} failed: {exc}")

    if ENSEMBLE is not None:
        ensemble_pred = float(np.asarray(ENSEMBLE.predict(X)).reshape(-1)[0])
    else:
        ensemble_pred = float(np.mean(list(individual.values()))) if individual else 0.0

    if individual:
        arr = np.fromiter(individual.values(), dtype=float)
        std = float(arr.std())
        margin = 1.96 * std / np.sqrt(len(arr))
    else:
        margin = 0.0

    return PredictionResponse(
        individual_predictions=individual,
        ensemble_weighted=ensemble_pred,
        ensemble_weights=dict(ENSEMBLE_WEIGHTS),
        confidence_range={
            "lower": ensemble_pred - margin,
            "upper": ensemble_pred + margin,
            "margin": margin,
        },
        timestamp=datetime.now().isoformat(),
    )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(cars: List[CarData]) -> BatchPredictionResponse:
    predictions = [await predict(car) for car in cars]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
