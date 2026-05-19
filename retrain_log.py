"""Retrain models with log-transformed target to fix overestimation for cheap cars."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import (
    CATEGORICAL_FEATURES,
    CURRENT_YEAR,
    DATA_FILE,
    ENCODER_FILE,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    SCALER_FILE,
    TARGET,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Fixed hyperparameters (stronger regularization)
# --------------------------------------------------------------------------- #

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
# Data pipeline
# --------------------------------------------------------------------------- #

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)
    if "Year" in df.columns:
        df["Car_Age"] = CURRENT_YEAR - df["Year"]
    return df

def preprocess(df, fit=True):
    df_encoded = df.copy()
    if fit:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, ENCODER_FILE, compress=True)
    else:
        encoders = joblib.load(ENCODER_FILE)
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                df_encoded[col] = encoders[col].transform(df_encoded[col].astype(str))

    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df_encoded[all_features].astype(float)
    y = df_encoded[TARGET]

    if fit:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, SCALER_FILE, compress=True)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler, encoders

# --------------------------------------------------------------------------- #
# Weighted Ensemble
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
# Training with log-transformed target
# --------------------------------------------------------------------------- #

def evaluate_model(y_true, y_pred, model_name="Model"):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    safe_y = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / safe_y)) * 100
    print(f"\n{model_name} Performance Metrics:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  RMSE:     ${rmse:,.2f}")
    print(f"  MAE:      ${mae:,.2f}")
    print(f"  MAPE:     {mape:.2f}%")
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2), "mape": float(mape), "mse": float(mse)}

def train_with_log(name, factory, X_train, X_test, y_train, y_test, save_path):
    print(f"\n{'='*60}")
    print(f"Training {name} (with log-transformed target) ...")
    print(f"{'='*60}")

    # Log-transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    model = factory()
    model.fit(X_train, y_train_log)

    # Predict in log space, then transform back
    y_train_pred_log = model.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred_log = model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)

    print(f"\n[{name}] training-set:")
    evaluate_model(y_train, y_train_pred, f"{name} (Train)")

    print(f"\n[{name}] held-out test-set:")
    test_metrics = evaluate_model(y_test, y_test_pred, f"{name} (Test)")

    # CV in log space
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=cv, scoring="r2", n_jobs=-1)
    print(f"\n[{name}] 5-fold CV R2 (log space): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[{name}] saved -> {save_path}")

    return {"name": name, "model": model, "test_metrics": test_metrics, "cv_scores": cv_scores.tolist()}

# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #

def _make_linear():
    return LinearRegression()

def _make_ridge():
    return Ridge(alpha=1.0, random_state=RANDOM_STATE)

def _make_lasso():
    return Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=10_000)

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
    "Linear Regression": (_make_linear, MODELS_DIR / "linear_model.pkl"),
    "Ridge Regression": (_make_ridge, MODELS_DIR / "ridge_model.pkl"),
    "Lasso Regression": (_make_lasso, MODELS_DIR / "lasso_model.pkl"),
    "Support Vector Regression": (_make_svr, MODELS_DIR / "svr_model.pkl"),
    "Random Forest": (_make_random_forest, MODELS_DIR / "random_forest_model.pkl"),
    "Gradient Boosting": (_make_gradient_boosting, MODELS_DIR / "gradient_boosting_model.pkl"),
    "XGBoost": (_make_xgboost, MODELS_DIR / "xgboost_model.pkl"),
    "AdaBoost": (_make_adaboost, MODELS_DIR / "adaboost_model.pkl"),
    "MLP": (_make_mlp, MODELS_DIR / "mlp_model.pkl"),
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("\n" + "="*60)
    print("RETRAINING WITH LOG-TRANSFORMED TARGET")
    print("="*60)

    df = load_and_clean(DATA_FILE)
    print(f"Data: {len(df):,} rows")
    print(f"Price: mean=${df['Price'].mean():,.0f}, median=${df['Price'].median():,.0f}")

    X_train, X_test, y_train, y_test, scaler, encoder = preprocess(df, fit=True)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    results = {}
    for name, (factory, save_path) in MODEL_REGISTRY.items():
        results[name] = train_with_log(name, factory, X_train, X_test, y_train, y_test, save_path)

    # Ensemble
    val_r2_lookup = {name: float(np.mean(r["cv_scores"])) for name, r in results.items()}
    base_models = {name: r["model"] for name, r in results.items()}
    ensemble = WeightedEnsemble(base_models, val_r2_lookup)

    print(f"\n{'='*60}")
    print("Ensemble weights")
    print(f"{'='*60}")
    for name, w in sorted(ensemble.get_weights().items(), key=lambda kv: -kv[1]):
        print(f"  {name:30s}  R2_val = {val_r2_lookup[name]:.4f}   w = {w:.4f}")

    # Ensemble test (need to predict in log space then transform back)
    y_test_pred_log = ensemble.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)
    ensemble_test_metrics = evaluate_model(y_test, y_test_pred, "Weighted Ensemble (Test)")

    # Save artifacts
    per_fold = {name: r["cv_scores"] for name, r in results.items()}
    metrics = {name: r["test_metrics"] for name, r in results.items()}
    metrics["Weighted Ensemble"] = ensemble_test_metrics
    weights = ensemble.get_weights()

    (MODELS_DIR / "per_fold_r2.json").write_text(json.dumps(per_fold, indent=2), encoding="utf-8")
    (MODELS_DIR / "all_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODELS_DIR / "ensemble_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")

    # Summary table
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    summary = pd.DataFrame([
        {"model": name, **m, "cv_mean_r2": float(np.mean(per_fold[name])), "cv_std_r2": float(np.std(per_fold[name]))}
        for name, m in metrics.items() if name in per_fold
    ]).sort_values("r2", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ----------------------------------------------------------------------- #
    # Ford Fiesta test case
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*60}")
    print("FORD FIESTA TEST CASE")
    print(f"{'='*60}")
    print("Input: FORD Fiesta, 2015, 90,000 mi, 1.2L, Petrol, Automatic, Hatchback")
    print("Expected: ~$5,000-$8,000")

    test_input = {
        "Brand": "FORD",
        "Model": "Fiesta",
        "Year": 2015,
        "Engine Size": 1.2,
        "Mileage": 90000,
        "Fuel Type": "Petrol",
        "Transmission": "Automatic",
        "Condition": "Hatchback",
    }

    test_df = pd.DataFrame([test_input])
    test_df["Car_Age"] = CURRENT_YEAR - test_df["Year"]

    for col in CATEGORICAL_FEATURES:
        if col in encoder:
            test_df[col] = encoder[col].transform(test_df[col].astype(str))

    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_test_input = test_df[all_features].astype(float)
    X_test_scaled = scaler.transform(X_test_input)

    print("\nIndividual model predictions:")
    predictions = {}
    for name, (factory, save_path) in MODEL_REGISTRY.items():
        try:
            model = joblib.load(save_path)
            pred_log = float(model.predict(X_test_scaled)[0])
            pred = float(np.expm1(pred_log))
            pred = max(0, pred)
            predictions[name] = pred
            print(f"  {name:30s}  ${pred:,.0f}")
        except Exception as e:
            print(f"  {name:30s}  Error: {e}")

    # Ensemble prediction (in log space, then transform back)
    ensemble_pred_log = float(ensemble.predict(X_test_scaled)[0])
    ensemble_pred = float(np.expm1(ensemble_pred_log))
    ensemble_pred = max(0, ensemble_pred)
    print(f"\n  {'Weighted Ensemble':30s}  ${ensemble_pred:,.0f}")

    preds = list(predictions.values())
    print(f"\n  Average:  ${np.mean(preds):,.0f}")
    print(f"  Std Dev:  ${np.std(preds):,.0f}")
    print(f"  Min:      ${np.min(preds):,.0f}")
    print(f"  Max:      ${np.max(preds):,.0f}")
    print(f"  Range:    ${np.max(preds) - np.min(preds):,.0f}")

    if 5000 <= ensemble_pred <= 8000:
        print(f"\n  SUCCESS: ${ensemble_pred:,.0f} is in expected range ($5,000-$8,000)")
    elif ensemble_pred < 10000:
        print(f"\n  PARTIAL: ${ensemble_pred:,.0f} is close to expected range")
    else:
        print(f"\n  WARNING: ${ensemble_pred:,.0f} is still overestimated")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
