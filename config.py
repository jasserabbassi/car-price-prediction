"""Configuration file for car price prediction model."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file
DATA_FILE = DATA_DIR / "car_price_prediction_.csv"

# Model paths
MODEL_RANDOM_FOREST = MODELS_DIR / "random_forest_model.pkl"
MODEL_XGBOOST = MODELS_DIR / "xgboost_model.pkl"
MODEL_GRADIENT_BOOSTING = MODELS_DIR / "gradient_boosting_model.pkl"
MODEL_NEURAL_NETWORK = MODELS_DIR / "neural_network_model.h5"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODER_FILE = MODELS_DIR / "encoder.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Feature engineering
CATEGORICAL_FEATURES = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
NUMERICAL_FEATURES = ['Year', 'Engine Size', 'Mileage']
TARGET = 'Price'

# Model hyperparameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'max_features': 'sqrt'
}

XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

GB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'subsample': 0.8,
    'random_state': RANDOM_STATE
}

# Neural network parameters
NN_PARAMS = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001
}
