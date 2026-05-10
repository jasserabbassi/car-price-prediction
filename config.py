"""Configuration file for the car price prediction project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = MODELS_DIR

# Data file
# Default to the cleaned real Kaggle dataset; fall back to the legacy synthetic CSV
# if the cleaned file is not present (so the repo still runs out of the box).
DATA_FILE_REAL = DATA_DIR / "car_price_kaggle_clean.csv"
DATA_FILE_LEGACY = DATA_DIR / "car_price_prediction_.csv"
DATA_FILE = DATA_FILE_REAL if DATA_FILE_REAL.exists() else DATA_FILE_LEGACY

# Model paths (one per algorithm)
MODEL_LINEAR = MODELS_DIR / "linear_model.pkl"
MODEL_RIDGE = MODELS_DIR / "ridge_model.pkl"
MODEL_LASSO = MODELS_DIR / "lasso_model.pkl"
MODEL_SVR = MODELS_DIR / "svr_model.pkl"
MODEL_RANDOM_FOREST = MODELS_DIR / "random_forest_model.pkl"
MODEL_GRADIENT_BOOSTING = MODELS_DIR / "gradient_boosting_model.pkl"
MODEL_XGBOOST = MODELS_DIR / "xgboost_model.pkl"
MODEL_ADABOOST = MODELS_DIR / "adaboost_model.pkl"
MODEL_MLP = MODELS_DIR / "mlp_model.pkl"
MODEL_NEURAL_NETWORK = MODELS_DIR / "mlp_model.pkl"  # alias kept for backward compat
MODEL_ENSEMBLE = MODELS_DIR / "weighted_ensemble.pkl"

SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODER_FILE = MODELS_DIR / "encoder.pkl"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15
CV_SPLITS = 5
CURRENT_YEAR = 2024  # used to compute Car_Age = CURRENT_YEAR - Year

# Feature engineering
NUMERICAL_FEATURES = ["Year", "Car_Age", "Engine Size", "Mileage"]
CATEGORICAL_FEATURES = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]
TARGET = "Price"

# Hyperparameters per model. tune_hyperparameters.py can overwrite this block.
LINEAR_PARAMS: dict = {}
RIDGE_PARAMS: dict = {"alpha": 1.0, "random_state": RANDOM_STATE}
LASSO_PARAMS: dict = {"alpha": 0.1, "random_state": RANDOM_STATE, "max_iter": 10_000}
SVR_PARAMS: dict = {"kernel": "rbf", "C": 10.0, "gamma": "scale"}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "max_features": "sqrt",
}

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}

ADABOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 32,
    "learning_rate_init": 1e-3,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 10,
    "random_state": RANDOM_STATE,
}
