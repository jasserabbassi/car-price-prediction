"""
FIXED model training script with proper preprocessing and XGBoost tuning.

Changes from original:
1. Uses ColumnTransformer (One-Hot + StandardScale)
2. No more LabelEncoding (fixes ordinal bias)
3. Improved XGBoost hyperparameters (fixes underfitting)
4. Better feature engineering
5. Validation on proper train/test split
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from dataclasses import dataclass
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
from xgboost import XGBRegressor

from config import (
    CV_SPLITS,
    DATA_FILE,
    RANDOM_STATE,
    MODELS_DIR,
)
from preprocess_fixed import (
    preprocess_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PREPROCESSOR_FILE,
    FEATURE_NAMES_FILE,
)
from utils import evaluate_model

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

# IMPROVED hyperparameters (tuned to fix underfitting)

LINEAR_PARAMS = {}

RIDGE_PARAMS = {
    "alpha": 0.5,
    "random_state": RANDOM_STATE,
}

LASSO_PARAMS = {
    "alpha": 0.05,
    "random_state": RANDOM_STATE,
    "max_iter": 10_000,
}

SVR_PARAMS = {
    "kernel": "rbf",
    "C": 100.0,  # Increased from 10 (less regularization, more flexibility)
    "gamma": "scale",
    "epsilon": 100,  # Tolerance band (car prices are large numbers)
}

RF_PARAMS = {
    "n_estimators": 300,  # Increased from 200
    "max_depth": 25,  # Increased from 20 (deeper trees)
    "min_samples_split": 3,  # Decreased from 5 (more splits)
    "min_samples_leaf": 1,  # Decreased from 2 (more flexibility)
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "max_features": "sqrt",
}

# XGBoost: Fixed to prevent underfitting
XGB_PARAMS = {
    "n_estimators": 500,  # Increased from 200 (more trees)
    "learning_rate": 0.1,  # Increased from 0.05 (larger steps)
    "max_depth": 10,  # Increased from 7 (deeper trees for complex relationships)
    "min_child_weight": 1,  # Lower = more flexible
    "subsample": 0.9,  # Increased from 0.8 (more data per tree)
    "colsample_bytree": 0.9,  # Increased from 0.8 (more features per tree)
    "colsample_bylevel": 0.9,
    "reg_alpha": 0.1,  # L1 regularization (light)
    "reg_lambda": 1.0,  # L2 regularization (prevents huge predictions)
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "objective": "reg:squarederror",
}

GB_PARAMS = {
    "n_estimators": 300,  # Increased from 200
    "learning_rate": 0.1,  # Increased from 0.05
    "max_depth": 8,  # Increased from 5
    "min_samples_split": 3,  # Decreased from 5
    "min_samples_leaf": 1,  # Decreased from 2
    "subsample": 0.9,  # Increased from 0.8
    "random_state": RANDOM_STATE,
}

ADABOOST_PARAMS = {
    "n_estimators": 200,  # Increased from 100
    "learning_rate": 0.15,  # Increased from 0.1
    "random_state": RANDOM_STATE,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64, 32),  # Deeper network
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-5,  # Lighter regularization
    "batch_size": 16,  # Smaller batch (more updates)
    "learning_rate_init": 5e-4,  # Lower initial rate (more stable)
    "max_iter": 500,  # More iterations
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,  # More patience
    "random_state": RANDOM_STATE,
}


# ============================================================================
# Weighted Ensemble (same as original, but with fixed data)
# ============================================================================

class WeightedEnsemble:
    """Performance-weighted ensemble of base models."""

    def __init__(
        self,
        models: Dict[str, object],
        val_r2_scores: Dict[str, float],
        top_k: int | None = 3,
    ):
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
        
        self.models = {k: models[k] for k in self.weights.keys()}
    
    def predict(self, X) -> np.ndarray:
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
    
    def get_weights(self) -> Dict[str, float]:
        return dict(self.weights)


WeightedEnsemble.__module__ = "train_model_fixed"


# ============================================================================
# Training Infrastructure
# ============================================================================

@dataclass
class TrainedModel:
    name: str
    model: object
    test_metrics: dict
    cv_scores: list


def _print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _save_pickle(model, path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path, compress=True)


def _train_one(
    name: str,
    factory: Callable[[], object],
    X_train,
    X_test,
    y_train,
    y_test,
    save_path,
) -> TrainedModel:
    """Train and evaluate a single model."""
    _print_banner(f"Training {name} …")
    
    model = factory()
    model.fit(X_train, y_train)
    
    # Evaluate on train set
    print(f"\n[{name}] Training set:")
    evaluate_model(y_train, model.predict(X_train), f"{name} (Train)")
    
    # Evaluate on test set
    print(f"\n[{name}] Test set:")
    test_metrics = evaluate_model(y_test, model.predict(X_test), f"{name} (Test)")
    
    # 5-fold cross-validation
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    print(f"\n[{name}] 5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"[{name}] Per-fold: {[round(float(x), 4) for x in cv_scores]}")
    
    _save_pickle(model, save_path)
    print(f"\n✓ [{name}] Saved to {save_path}")
    
    return TrainedModel(name, model, test_metrics, [float(x) for x in cv_scores])


# ============================================================================
# Model Factories
# ============================================================================

def _make_linear():
    return LinearRegression(**LINEAR_PARAMS)

def _make_ridge():
    return Ridge(**RIDGE_PARAMS)

def _make_lasso():
    return Lasso(**LASSO_PARAMS)

def _make_svr():
    return SVR(**SVR_PARAMS)

def _make_random_forest():
    return RandomForestRegressor(**RF_PARAMS)

def _make_gradient_boosting():
    return GradientBoostingRegressor(**GB_PARAMS)

def _make_xgboost():
    return XGBRegressor(**XGB_PARAMS)

def _make_adaboost():
    return AdaBoostRegressor(**ADABOOST_PARAMS)

def _make_mlp():
    return MLPRegressor(**MLP_PARAMS)


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    _print_banner("CAR PRICE PREDICTOR - FIXED TRAINING PIPELINE")
    print("\nKey fixes applied:")
    print("  ✓ One-Hot Encoding (not LabelEncoding)")
    print("  ✓ StandardScaler only on numerical features")
    print("  ✓ Improved XGBoost hyperparameters")
    print("  ✓ Better feature engineering")
    print("  ✓ No data leakage in train/test split")
    
    # ========== LOAD & PREPROCESS DATA ==========
    print("\n" + "-" * 70)
    print("STEP 1: Load & Preprocess Data")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        Path(DATA_FILE),
        fit=True,
    )
    
    print(f"\n✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Features: {len(feature_names)}")
    
    # ========== TRAIN MODELS ==========
    print("\n" + "-" * 70)
    print("STEP 2: Train 9 Base Models")
    print("-" * 70)
    
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(exist_ok=True)
    
    trained_models = []
    
    # Train each model
    trained_models.append(_train_one(
        "Linear Regression",
        _make_linear,
        X_train, X_test, y_train, y_test,
        models_dir / "linear_model_fixed.pkl",
    ))
    
    trained_models.append(_train_one(
        "Ridge Regression",
        _make_ridge,
        X_train, X_test, y_train, y_test,
        models_dir / "ridge_model_fixed.pkl",
    ))
    
    trained_models.append(_train_one(
        "Lasso Regression",
        _make_lasso,
        X_train, X_test, y_train, y_test,
        models_dir / "lasso_model_fixed.pkl",
    ))
    
    trained_models.append(_train_one(
        "SVR",
        _make_svr,
        X_train, X_test, y_train, y_test,
        models_dir / "svr_model_fixed.pkl",
    ))
    
    trained_models.append(_train_one(
        "Random Forest",
        _make_random_forest,
        X_train, X_test, y_train, y_test,
        models_dir / "random_forest_model_fixed.pkl",
    ))
    
    trained_models.append(_train_one(
        "Gradient Boosting",
        _make_gradient_boosting,
        X_train, X_test, y_train, y_test,
        models_dir / "gradient_boosting_model_fixed.pkl",
    ))
    
    trained_models.append(_train_one(
        "XGBoost",
        _make_xgboost,
        X_train, X_test, y_train, y_test,
        models_dir / "xgboost_model_fixed.pkl",
    ))
    
    trained_models.append(_train_one(
        "AdaBoost",
        _make_adaboost,
        X_train, X_test, y_train, y_test,
        models_dir / "adaboost_model_fixed.pkl",
    ))
    
    trained_models.append(_train_one(
        "MLP",
        _make_mlp,
        X_train, X_test, y_train, y_test,
        models_dir / "mlp_model_fixed.pkl",
    ))
    
    # ========== BUILD WEIGHTED ENSEMBLE ==========
    print("\n" + "-" * 70)
    print("STEP 3: Build Weighted Ensemble")
    print("-" * 70)
    
    base_models = {m.name: joblib.load(models_dir / f"{m.name.lower().replace(' ', '_')}_model_fixed.pkl")
                   for m in trained_models}
    
    val_r2_scores = {m.name: np.mean(m.cv_scores) for m in trained_models}
    
    _print_banner("Test Set Performance Summary")
    print("\nModel\t\t\t\tR²\t\tRMSE\t\tMAE")
    print("-" * 70)
    for m in trained_models:
        r2 = m.test_metrics["r2"]
        rmse = m.test_metrics["rmse"]
        mae = m.test_metrics["mae"]
        print(f"{m.name:30s}\t{r2:.4f}\t\t${rmse:,.0f}\t\t${mae:,.0f}")
    
    # Create ensemble
    ensemble = WeightedEnsemble(base_models, val_r2_scores, top_k=3)
    
    print(f"\nEnsemble weights (top-3):")
    for name, weight in sorted(ensemble.get_weights().items(), key=lambda x: -x[1]):
        print(f"  {name:30s}: {weight:.4f}")
    
    # Evaluate ensemble on test set
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_metrics = evaluate_model(y_test, y_pred_ensemble, "Weighted Ensemble (Test)")
    
    # Save ensemble
    _save_pickle(ensemble, models_dir / "weighted_ensemble_fixed.pkl")
    
    # ========== SAVE METADATA ==========
    print("\n" + "-" * 70)
    print("STEP 4: Save Metadata")
    print("-" * 70)
    
    # Save weights
    weights_dict = ensemble.get_weights()
    with open(models_dir / "ensemble_weights_fixed.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"✓ Saved ensemble weights")
    
    # Save all metrics
    all_metrics = {m.name: m.test_metrics for m in trained_models}
    all_metrics["Weighted Ensemble"] = ensemble_metrics
    
    with open(models_dir / "all_metrics_fixed.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Saved all metrics")
    
    # Save per-fold R² scores
    per_fold_r2 = {m.name: m.cv_scores for m in trained_models}
    with open(models_dir / "per_fold_r2_fixed.json", "w") as f:
        json.dump(per_fold_r2, f, indent=2)
    print(f"✓ Saved per-fold R² scores")
    
    _print_banner("TRAINING COMPLETE!")
    print("\nOutput files:")
    print(f"  Models: {models_dir}/*.pkl")
    print(f"  Preprocessor: {PREPROCESSOR_FILE}")
    print(f"  Feature names: {FEATURE_NAMES_FILE}")
    print(f"  Weights: {models_dir}/ensemble_weights_fixed.json")
    print(f"  Metrics: {models_dir}/all_metrics_fixed.json")


if __name__ == "__main__":
    main()
