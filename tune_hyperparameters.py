"""Hyperparameter tuning via RandomizedSearchCV (Bergstra & Bengio, 2012).

By default this performs 20 search iterations × 5-fold CV per model, scoring
on R². With --update-config it overwrites the corresponding *_PARAMS dicts in
config.py so subsequent runs of train_model.py use the tuned values.

    python tune_hyperparameters.py                # tune + write JSON, leave config.py untouched
    python tune_hyperparameters.py --update-config  # also rewrite the *_PARAMS dicts
    python tune_hyperparameters.py --models XGBoost RandomForest  # tune only some
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from config import (
    CV_SPLITS,
    DATA_FILE,
    MODELS_DIR,
    RANDOM_STATE,
)
from utils import preprocess_data

warnings.filterwarnings("ignore")

PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "RandomForest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    },
    "MLP": {
        "hidden_layer_sizes": [(64,), (128,), (128, 64), (128, 64, 32)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-4, 1e-3, 5e-3],
        "batch_size": [16, 32, 64],
    },
}

ESTIMATOR_FACTORIES = {
    "RandomForest": lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": lambda: XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    "GradientBoosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
    "MLP": lambda: MLPRegressor(
        random_state=RANDOM_STATE,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
    ),
}

# How tuned param dicts map back to the *_PARAMS variable names in config.py.
CONFIG_NAMES = {
    "RandomForest": "RF_PARAMS",
    "XGBoost": "XGB_PARAMS",
    "GradientBoosting": "GB_PARAMS",
    "MLP": "MLP_PARAMS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(PARAM_GRIDS.keys()),
        default=sorted(PARAM_GRIDS.keys()),
        help="subset of models to tune (default: all)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="number of RandomizedSearchCV iterations per model (default: 20)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=CV_SPLITS,
        help=f"K for K-fold CV (default: {CV_SPLITS})",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="rewrite the *_PARAMS dicts in config.py with the best params",
    )
    return parser.parse_args()


def tune(name: str, X_train, y_train, n_iter: int, cv: int) -> dict:
    print(f"\n[{name}] running RandomizedSearchCV (n_iter={n_iter}, cv={cv}) …")
    estimator = ESTIMATOR_FACTORIES[name]()
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_GRIDS[name],
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"[{name}] best CV R² = {search.best_score_:.4f}")
    print(f"[{name}] best params: {search.best_params_}")
    return {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "n_iter": n_iter,
        "cv": cv,
    }


def update_config_file(best_params_by_model: Dict[str, dict]) -> None:
    """Replace the *_PARAMS dict literals in config.py with the tuned values."""
    config_path = Path(__file__).parent / "config.py"
    text = config_path.read_text()

    for model_name, payload in best_params_by_model.items():
        var_name = CONFIG_NAMES.get(model_name)
        if var_name is None:
            continue

        params = dict(payload["best_params"])
        # Always preserve random_state where applicable.
        if model_name in {"RandomForest", "XGBoost", "GradientBoosting"}:
            params.setdefault("random_state", RANDOM_STATE)
        if model_name == "RandomForest":
            params.setdefault("n_jobs", -1)
        if model_name == "XGBoost":
            params.setdefault("verbosity", 0)
        if model_name == "MLP":
            params.setdefault("random_state", RANDOM_STATE)
            params.setdefault("max_iter", 300)
            params.setdefault("early_stopping", True)
            params.setdefault("n_iter_no_change", 10)

        formatted = "{\n" + "".join(f"    {repr(k)}: {repr(v)},\n" for k, v in params.items()) + "}"
        pattern = rf"^{var_name}\s*=\s*\{{[^{{}}]*\}}"
        replacement = f"{var_name} = {formatted}"
        new_text, n_subs = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
        if n_subs == 0:
            print(f"[update-config] WARNING: could not locate {var_name} in config.py")
        else:
            text = new_text
            print(f"[update-config] rewrote {var_name}")

    config_path.write_text(text, encoding="utf-8")
    print(f"[update-config] saved {config_path}")


def main() -> int:
    args = parse_args()
    print(f"Loading data from {DATA_FILE}")
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(DATA_FILE, fit=True)
    print(f"X_train shape: {X_train.shape}")

    results = {}
    for name in args.models:
        results[name] = tune(name, X_train, y_train, args.n_iter, args.cv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "best_hyperparameters.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[tune] saved {out_path}")

    if args.update_config:
        update_config_file(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
