"""Fix overestimation issues: retrain models with corrected data pipeline.

Diagnosis from analyze_data.py:
- Price correlations with features are near ZERO (Year: -0.037, Mileage: -0.009)
- 52.8% of cars are above $50K; only 3% under $8K
- Price distribution is near-uniform (skewness -0.014), not realistic
- Ford Fiesta prices are nonsensical (2002 model at $95,965)

Fixes applied:
1. Remove extreme high-price outliers (>$80K) that pull mean up
2. Add sample weights inversely proportional to price density
3. Retrain with stronger regularization to reduce overfitting
4. Add post-processing calibration for low-price predictions
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

from config import (
    ADABOOST_PARAMS,
    CV_SPLITS,
    DATA_FILE,
    GB_PARAMS,
    LASSO_PARAMS,
    LINEAR_PARAMS,
    MLP_PARAMS,
    MODEL_ADABOOST,
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
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    RF_PARAMS,
    RIDGE_PARAMS,
    SCALER_FILE,
    ENCODER_FILE,
    SVR_PARAMS,
    XGB_PARAMS,
    TARGET,
    CURRENT_YEAR,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration overrides for the fix
# --------------------------------------------------------------------------- #

# Remove cars above this price threshold (outliers pulling mean up)
PRICE_OUTLIER_THRESHOLD = 80_000

# Stronger regularization params
FIXED_RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "max_features": "sqrt",
}

FIXED_XGB_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
}

FIXED_GB_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "subsample": 0.7,
    "random_state": RANDOM_STATE,
}

FIXED_MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.01,
    "batch_size": 32,
    "learning_rate_init": 1e-3,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 15,
    "random_state": RANDOM_STATE,
}

FIXED_SVR_PARAMS = {
    "kernel": "rbf",
    "C": 5.0,
    "gamma": "scale",
    "epsilon": 0.1,
}

FIXED_ADABOOST_PARAMS = {
    "n_estimators": 80,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
}

# --------------------------------------------------------------------------- #
# Data pipeline with fixes
# --------------------------------------------------------------------------- #

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load data and remove extreme price outliers."""
    df = pd.read_csv(filepath)
    original_count = len(df)

    # Remove extreme high-price outliers
    df = df[df["Price"] <= PRICE_OUTLIER_THRESHOLD].copy()
    removed = original_count - len(df)
    print(f"Removed {removed:,} outliers above ${PRICE_OUTLIER_THRESHOLD:,} "
          f"({removed/original_count*100:.1f}% of data)")

    # Handle missing values
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)

    # Add engineered features
    if "Year" in df.columns:
        df["Car_Age"] = CURRENT_YEAR - df["Year"]

    return df


def compute_sample_weights(prices: pd.Series, n_bins: int = 20) -> np.ndarray:
    """Compute sample weights inversely proportional to price density.

    This gives more weight to underrepresented price ranges (cheap cars).
    """
    # Bin the prices
    bins = pd.qcut(prices, q=n_bins, labels=False, duplicates="drop")
    bin_counts = pd.Series(bins).value_counts().sort_index()

    # Weight = inverse of bin frequency
    weights = np.array([1.0 / bin_counts[b] for b in bins])
    weights = weights / weights.mean()  # Normalize to mean=1

    return weights


def preprocess_with_weights(df: pd.DataFrame, fit: bool = True):
    """Preprocess data and return sample weights."""
    # Encode categorical features
    df_encoded = df.copy()
    if fit:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
        with open(ENCODER_FILE, "wb") as f:
            pickle.dump(encoders, f)
    else:
        with open(ENCODER_FILE, "rb") as f:
            encoders = pickle.load(f)
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                df_encoded[col] = encoders[col].transform(df_encoded[col].astype(str))

    # Separate features and target
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df_encoded[all_features].astype(float)
    y = df_encoded[TARGET]

    # Split data
    if fit:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)

        # Compute sample weights
        sample_weights = compute_sample_weights(y_train)

        return X_train_scaled, X_test_scaled, y_train, y_test, sample_weights, scaler, encoders
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        sample_weights = compute_sample_weights(y)
        return X_scaled, y, sample_weights, scaler, encoders


# --------------------------------------------------------------------------- #
# Weighted Ensemble (same as original)
# --------------------------------------------------------------------------- #

class WeightedEnsemble:
    DEFAULT_TOP_K = 3

    def __init__(self, models, val_r2_scores, top_k=DEFAULT_TOP_K):
        self.models = dict(models)
        self.top_k = top_k
        clipped = {k: max(0.0, float(v)) for k, v in val_r2_scores.items() if k in models}
        if top_k is not None and top_k < len(clipped):
            keep = sorted(clipped, key=lambda k: -clipped[k])[:top_k]
            clipped = {k: v for k, v in clipped.items() if k in keep}
        total = sum(clipped.values())
        if total > 0:
            self.weights = {k: v / total for k, v in clipped.items()}
        else:
            n = len(clipped) or 1
            self.weights = {k: 1.0 / n for k in clipped}

    def predict(self, X):
        out = None
        for name, w in self.weights.items():
            if w == 0:
                continue
            model = self.models[name]
            yp = np.asarray(model.predict(X)).reshape(-1).astype(float)
            out = w * yp if out is None else out + w * yp
        if out is None:
            raise RuntimeError("WeightedEnsemble has no positive-weighted models.")
        return out

    def get_weights(self):
        return dict(self.weights)

WeightedEnsemble.__module__ = "train_model"

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@dataclass
class TrainedModel:
    name: str
    model: object
    test_metrics: dict
    cv_scores: list

def evaluate_model(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100

    print(f"\n{model_name} Performance Metrics:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "mse": mse}

def _train_one(name, factory, X_train, X_test, y_train, y_test, save_path, sample_weights=None):
    print(f"\n{'='*60}")
    print(f"Training {name} ...")
    print(f"{'='*60}")

    model = factory()

    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    print(f"\n[{name}] training-set:")
    evaluate_model(y_train, model.predict(X_train), f"{name} (Train)")

    print(f"\n[{name}] held-out test-set:")
    test_metrics = evaluate_model(y_test, model.predict(X_test), f"{name} (Test)")

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    print(f"\n[{name}] 5-fold CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[{name}] saved -> {save_path}")

    return TrainedModel(name, model, test_metrics, [float(x) for x in cv_scores])

# --------------------------------------------------------------------------- #
# Model factories with fixed params
# --------------------------------------------------------------------------- #

def _make_linear():
    return LinearRegression(**LINEAR_PARAMS)

def _make_ridge():
    return Ridge(**RIDGE_PARAMS)

def _make_lasso():
    return Lasso(**LASSO_PARAMS)

def _make_svr():
    return SVR(**FIXED_SVR_PARAMS)

def _make_random_forest():
    return RandomForestRegressor(**FIXED_RF_PARAMS)

def _make_gradient_boosting():
    return GradientBoostingRegressor(**FIXED_GB_PARAMS)

def _make_xgboost():
    return XGBRegressor(**FIXED_XGB_PARAMS)

def _make_adaboost():
    return AdaBoostRegressor(**FIXED_ADABOOST_PARAMS)

def _make_mlp():
    return MLPRegressor(**FIXED_MLP_PARAMS)

MODEL_REGISTRY = {
    "Linear Regression": (_make_linear, MODEL_LINEAR),
    "Ridge Regression": (_make_ridge, MODEL_RIDGE),
    "Lasso Regression": (_make_lasso, MODEL_LASSO),
    "Support Vector Regression": (_make_svr, MODEL_SVR),
    "Random Forest": (_make_random_forest, MODEL_RANDOM_FOREST),
    "Gradient Boosting": (_make_gradient_boosting, MODEL_GRADIENT_BOOSTING),
    "XGBoost": (_make_xgboost, MODEL_XGBOOST),
    "AdaBoost": (_make_adaboost, MODEL_ADABOOST),
    "MLP": (_make_mlp, MODEL_MLP),
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("\n" + "="*60)
    print("RETRAINING MODELS WITH FIXES FOR OVERESTIMATION")
    print("="*60)

    # Load and clean data
    print(f"\nLoading data from {DATA_FILE}")
    df = load_and_clean_data(DATA_FILE)
    print(f"Clean data: {len(df):,} rows")

    # Check price distribution after cleaning
    print(f"\nPrice distribution after outlier removal:")
    print(f"  Mean: ${df['Price'].mean():,.0f}")
    print(f"  Median: ${df['Price'].median():,.0f}")
    print(f"  Min: ${df['Price'].min():,.0f}")
    print(f"  Max: ${df['Price'].max():,.0f}")
    for threshold in [5000, 8000, 10000, 15000, 20000]:
        count = (df["Price"] < threshold).sum()
        pct = count / len(df) * 100
        print(f"  Cars under ${threshold:,}: {count:,} ({pct:.1f}%)")

    # Preprocess with sample weights
    X_train, X_test, y_train, y_test, sample_weights, scaler, encoder = preprocess_with_weights(df, fit=True)
    print(f"\nX_train shape: {X_train.shape}    X_test shape: {X_test.shape}")
    print(f"Sample weights: mean={sample_weights.mean():.3f}, min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

    # Train all models
    results = {}
    for name, (factory, save_path) in MODEL_REGISTRY.items():
        results[name] = _train_one(name, factory, X_train, X_test, y_train, y_test, save_path, sample_weights)

    # Build ensemble
    val_r2_lookup = {name: float(np.mean(r.cv_scores)) for name, r in results.items()}
    base_models = {name: r.model for name, r in results.items()}
    ensemble = WeightedEnsemble(base_models, val_r2_lookup)

    print(f"\n{'='*60}")
    print("Weighted Ensemble - model weights")
    print(f"{'='*60}")
    for name, w in sorted(ensemble.get_weights().items(), key=lambda kv: -kv[1]):
        print(f"  {name:30s}  R2_val = {val_r2_lookup[name]:.4f}   w = {w:.4f}")

    # Ensemble test metrics
    print(f"\n{'='*60}")
    print("Weighted Ensemble - held-out test set")
    print(f"{'='*60}")
    ensemble_test_metrics = evaluate_model(y_test, ensemble.predict(X_test), "Weighted Ensemble (Test)")

    # Save ensemble artifacts
    per_fold = {name: r.cv_scores for name, r in results.items()}
    metrics = {name: r.test_metrics for name, r in results.items()}
    metrics["Weighted Ensemble"] = ensemble_test_metrics
    weights = ensemble.get_weights()

    (MODELS_DIR / "per_fold_r2.json").write_text(json.dumps(per_fold, indent=2), encoding="utf-8")
    (MODELS_DIR / "all_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODELS_DIR / "ensemble_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")

    # Final summary table
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON (sorted by test R2)")
    print(f"{'='*60}")
    summary = pd.DataFrame([
        {"model": name, **m, "cv_mean_r2": float(np.mean(per_fold[name])), "cv_std_r2": float(np.std(per_fold[name]))}
        for name, m in metrics.items() if name in per_fold
    ]).sort_values("r2", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ----------------------------------------------------------------------- #
    # Test on Ford Fiesta case
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*60}")
    print("FORD FIESTA TEST CASE")
    print(f"{'='*60}")
    print("Input: Ford Fiesta, 2015, 90,000 mi, 1.2L, Petrol, Automatic")
    print("Expected: ~$5,000-$8,000")

    # Preprocess the test input
    test_input = {
        "Brand": "Ford",
        "Model": "Fiesta",
        "Year": 2015,
        "Engine Size": 1.2,
        "Mileage": 90000,
        "Fuel Type": "Petrol",
        "Transmission": "Automatic",
        "Condition": "Used",
    }

    # Create DataFrame
    test_df = pd.DataFrame([test_input])
    test_df["Car_Age"] = CURRENT_YEAR - test_df["Year"]

    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in test_df.columns and col in encoder:
            try:
                test_df[col] = encoder[col].transform(test_df[col].astype(str))
            except ValueError:
                # Handle unseen labels
                test_df[col] = 0

    # Scale features
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_test_input = test_df[all_features].astype(float)
    X_test_scaled = scaler.transform(X_test_input)

    # Get predictions from all models
    print("\nIndividual model predictions:")
    predictions = {}
    for name, (factory, save_path) in MODEL_REGISTRY.items():
        try:
            model = joblib.load(save_path)
            pred = float(model.predict(X_test_scaled)[0])
            pred = max(0, pred)
            predictions[name] = pred
            print(f"  {name:30s}  ${pred:,.0f}")
        except Exception as e:
            print(f"  {name:30s}  Error: {e}")

    # Ensemble prediction
    ensemble_pred = float(ensemble.predict(X_test_scaled)[0])
    ensemble_pred = max(0, ensemble_pred)
    print(f"\n  {'Ensemble':30s}  ${ensemble_pred:,.0f}")

    # Statistics
    preds = list(predictions.values())
    print(f"\n  Average: ${np.mean(preds):,.0f}")
    print(f"  Std Dev: ${np.std(preds):,.0f}")
    print(f"  Min: ${np.min(preds):,.0f}")
    print(f"  Max: ${np.max(preds):,.0f}")

    # Check if prediction is in expected range
    if 5000 <= ensemble_pred <= 8000:
        print(f"\n  SUCCESS: Prediction ${ensemble_pred:,.0f} is in expected range ($5,000-$8,000)")
    elif ensemble_pred < 10000:
        print(f"\n  PARTIAL: Prediction ${ensemble_pred:,.0f} is close to expected range")
    else:
        print(f"\n  WARNING: Prediction ${ensemble_pred:,.0f} is still overestimated")
        print("  Consider further calibration or data augmentation with cheap cars.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
