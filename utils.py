"""Utility functions for data preprocessing and model evaluation."""

from __future__ import annotations

import os
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    CURRENT_YEAR,
    ENCODER_FILE,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    SCALER_FILE,
    TARGET,
    TEST_SIZE,
)


def load_data(filepath) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    return pd.read_csv(filepath)


def add_engineered_features(df: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Add derived features used by the thesis (chapter 3 §2.4).

    Currently this computes Car_Age = current_year - Year. Year is kept as well
    so models that prefer the raw value can still use it; downstream code can
    drop one or the other via NUMERICAL_FEATURES in config.py.
    """
    df = df.copy()
    if "Year" in df.columns:
        df["Car_Age"] = (current_year - pd.to_numeric(df["Year"], errors="coerce")).astype(float)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation for numeric, mode imputation for categorical."""
    df = df.copy()
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            mode = df[col].mode()
            fill_val = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
    return df


def remove_outliers(df: pd.DataFrame, columns=None, threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows whose z-score on any of the given columns exceeds threshold."""
    if columns is None:
        columns = [c for c in NUMERICAL_FEATURES + [TARGET] if c in df.columns]

    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            data = df_clean[col].astype(float)
            std = data.std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
            z_scores = np.abs((data - data.mean()) / std)
            df_clean = df_clean[z_scores < threshold]
    return df_clean


def encode_categorical_features(df: pd.DataFrame, fit: bool = True, encoder=None):
    """Encode categorical columns with per-column LabelEncoders."""
    df_encoded = df.copy()

    if fit:
        encoders: dict[str, LabelEncoder] = {}
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le

        os.makedirs(os.path.dirname(ENCODER_FILE), exist_ok=True)
        joblib.dump(encoders, ENCODER_FILE, compress=True)

        return df_encoded, encoders

    if encoder is None:
        encoder = joblib.load(ENCODER_FILE)

    for col in CATEGORICAL_FEATURES:
        if col in df_encoded.columns:
            df_encoded[col] = encoder[col].transform(df_encoded[col].astype(str))

    return df_encoded, encoder


def scale_features(X_train, X_test, fit: bool = True, scaler=None):
    """Standard-scale features. Saves scaler to disk on first fit."""
    if fit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)
        joblib.dump(scaler, SCALER_FILE, compress=True)
        return X_train_scaled, X_test_scaled, scaler

    if scaler is None:
        scaler = joblib.load(SCALER_FILE)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(filepath, fit: bool = True, encoder=None, scaler=None):
    """Complete preprocessing pipeline used by `train_model.py`."""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = add_engineered_features(df)
    df = remove_outliers(df)
    df, encoder = encode_categorical_features(df, fit=fit, encoder=encoder)

    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    available = [f for f in all_features if f in df.columns]
    X = df[available].astype(float)
    y = df[TARGET]

    if fit:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_train, X_test, scaler = scale_features(X_train, X_test, fit=True)
        return X_train, X_test, y_train, y_test, scaler, encoder

    if scaler is None:
        scaler = joblib.load(SCALER_FILE)
    X_scaled, _, scaler = scale_features(X, X, fit=False, scaler=scaler)
    return X_scaled, scaler, encoder


def preprocess_input_data(input_dict, encoder, scaler):
    """Preprocess a single user-facing input row for online prediction."""
    try:
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        df = pd.DataFrame([input_dict])

        # Compute Car_Age if possible.
        if "Year" in df.columns and "Car_Age" not in df.columns:
            df["Car_Age"] = float(CURRENT_YEAR) - float(df["Year"].iloc[0])

        # Required-feature check.
        for feature in all_features:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")

        # Validate categorical values exist in encoder.
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col in encoder:
                value = str(df[col].iloc[0])
                valid_classes = list(encoder[col].classes_)
                if value not in valid_classes:
                    raise ValueError(
                        f"Invalid value '{value}' for {col}. "
                        f"Valid options: {valid_classes[:10]}{'…' if len(valid_classes) > 10 else ''}"
                    )

        # Encode categoricals.
        df_encoded = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns and col in encoder:
                df_encoded[col] = encoder[col].transform(df_encoded[col].astype(str))

        X = df_encoded[all_features].astype(float)
        if X.shape[1] != len(all_features):
            raise ValueError(
                f"Feature shape mismatch: expected {len(all_features)}, got {X.shape[1]}"
            )

        return scaler.transform(X)
    except Exception as exc:  # pragma: no cover - re-raised
        raise ValueError(f"Preprocessing error: {exc}") from exc


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """Compute and print the standard regression metrics."""
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    safe_y = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / safe_y)) * 100)

    print(f"\n{model_name} Performance Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE:     ${rmse:,.2f}")
    print(f"  MAE:      ${mae:,.2f}")
    print(f"  MAPE:     {mape:.2f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "mse": mse}


def get_feature_importance(model, feature_names):
    """Extract feature importance dict from tree-based estimators."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}
    return None
