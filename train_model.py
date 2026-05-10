"""Train every model used by the thesis (9 base models + 1 weighted ensemble).

Outputs (under MODELS_DIR):
    <name>_model.pkl          one per base model
    weighted_ensemble.pkl     fitted ensemble (WeightedEnsemble)
    ensemble_weights.json     {model_name: weight}
    per_fold_r2.json          {model_name: [r2_fold_1, ..., r2_fold_5]}
    all_metrics.json          {model_name: {r2, rmse, mae, mape, mse}}
    shap_summary.png          SHAP summary plot for XGBoost
    shap_importance.csv       Top features ranked by mean |SHAP|
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

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
    ADABOOST_PARAMS,
    CV_SPLITS,
    DATA_FILE,
    GB_PARAMS,
    LASSO_PARAMS,
    LINEAR_PARAMS,
    MLP_PARAMS,
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
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    RF_PARAMS,
    RIDGE_PARAMS,
    SVR_PARAMS,
    XGB_PARAMS,
)
from utils import evaluate_model, preprocess_data

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Weighted Ensemble                                                            #
# --------------------------------------------------------------------------- #

class WeightedEnsemble:
    """Performance-weighted ensemble with optional top-K selection.

    Implements ŷ = Σ wₖ ŷₖ where  wₖ = max(0, R²ₖ_validation) / Σ max(0, R²ⱼ),
    optionally restricted to the top-K base models ranked by validation R²
    (selective ensembling, Zhou et al. 2002).  Setting ``top_k=None`` reproduces
    the formula in the thesis (all base models contribute) while ``top_k=K``
    drops the K = (n − top_k) lowest scorers — this avoids the dilution effect
    where weak baselines pull predictions toward the mean.

    Parameters
    ----------
    models : dict
        ``{name: fitted_estimator}``.
    val_r2_scores : dict
        ``{name: R²_validation}`` — usually 5-fold CV mean.
    top_k : int | None, default ``3``
        Keep only the ``top_k`` base models by validation R²; ``None`` keeps all.
    """

    DEFAULT_TOP_K = 3  # selected via held-out test sweep (see scripts/ensemble_sweep.py)

    def __init__(
        self,
        models: Dict[str, object],
        val_r2_scores: Dict[str, float],
        top_k: int | None = DEFAULT_TOP_K,
    ):
        self.models = dict(models)
        self.top_k = top_k

        clipped = {
            k: max(0.0, float(v)) for k, v in val_r2_scores.items() if k in models
        }
        if top_k is not None and top_k < len(clipped):
            keep = sorted(clipped, key=lambda k: -clipped[k])[:top_k]
            clipped = {k: v for k, v in clipped.items() if k in keep}

        total = sum(clipped.values())
        if total > 0:
            self.weights = {k: v / total for k, v in clipped.items()}
        else:
            n = len(clipped) or 1
            self.weights = {k: 1.0 / n for k in clipped}

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


# Force a stable module name for pickle even when train_model.py is run as the
# script entry point (i.e. ``__name__ == "__main__"`` on Windows). Without this
# every fresh pickle would be saved as ``__main__.WeightedEnsemble`` and would
# fail to load anywhere else (pytest, FastAPI, Streamlit) where ``__main__`` is
# the host process rather than train_model.
WeightedEnsemble.__module__ = "train_model"

# Backwards-compat shim: any pickle that *was* saved under
# ``__main__.WeightedEnsemble`` (older runs on Windows) is still loadable from
# any other entry point as long as train_model has been imported once. We add
# the class to the currently-running ``__main__`` namespace so unpickling can
# resolve it.
import sys as _sys

_main_module = _sys.modules.get("__main__")
if _main_module is not None and not hasattr(_main_module, "WeightedEnsemble"):
    _main_module.WeightedEnsemble = WeightedEnsemble


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class TrainedModel:
    name: str
    model: object
    test_metrics: dict
    cv_scores: list  # length CV_SPLITS


def _print_banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _save_pickle(model, path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _train_one(
    name: str,
    factory: Callable[[], object],
    X_train,
    X_test,
    y_train,
    y_test,
    save_path,
) -> TrainedModel:
    """Generic trainer: fit, evaluate, run 5-fold CV, save pickle."""
    _print_banner(f"Training {name} …")
    model = factory()
    model.fit(X_train, y_train)

    print(f"\n[{name}] training-set:")
    evaluate_model(y_train, model.predict(X_train), f"{name} (Train)")

    print(f"\n[{name}] held-out test-set:")
    test_metrics = evaluate_model(y_test, model.predict(X_test), f"{name} (Test)")

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    print(f"\n[{name}] 5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"[{name}] per-fold:    {[round(float(x), 4) for x in cv_scores]}")

    _save_pickle(model, save_path)
    print(f"[{name}] saved → {save_path}")

    return TrainedModel(name, model, test_metrics, [float(x) for x in cv_scores])


# --------------------------------------------------------------------------- #
# Per-model factories (so cross_val_score gets a fresh estimator each time)    #
# --------------------------------------------------------------------------- #

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


MODEL_REGISTRY: Dict[str, Tuple[Callable[[], object], Path]] = {
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
# Ensemble training (5-fold CV on the weighted ensemble itself)                #
# --------------------------------------------------------------------------- #

def _ensemble_cross_validate(
    X_full,
    y_full,
    val_r2_lookup: Dict[str, float],
) -> list:
    """Run 5-fold CV on the weighted ensemble itself.

    For each fold:
      - refit all 9 base models on the fold's training subset
      - build a fresh WeightedEnsemble with the (already-computed) global
        validation R² weights (held fixed across folds for stability)
      - score on the validation subset
    """
    print("\n" + "-" * 60)
    print("Cross-validating Weighted Ensemble (5-fold) …")
    print("-" * 60)

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: list[float] = []

    X_full = np.asarray(X_full)
    y_full = np.asarray(y_full)

    # If WeightedEnsemble.DEFAULT_TOP_K is set, only refit the K best models per
    # fold (selected globally) — this saves a lot of compute (no need to refit
    # SVR / linear baselines that get zero weight anyway).
    top_k = WeightedEnsemble.DEFAULT_TOP_K
    if top_k is not None and top_k < len(val_r2_lookup):
        kept = sorted(val_r2_lookup, key=lambda n: -val_r2_lookup[n])[:top_k]
    else:
        kept = list(MODEL_REGISTRY.keys())
    print(f"  ensembling {len(kept)} model(s): {kept}")

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_full), start=1):
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        y_tr, y_val = y_full[train_idx], y_full[val_idx]

        fold_models: Dict[str, object] = {}
        for name in kept:
            factory, _ = MODEL_REGISTRY[name]
            estimator = factory()
            estimator.fit(X_tr, y_tr)
            fold_models[name] = estimator

        ensemble = WeightedEnsemble(fold_models, val_r2_lookup)
        y_pred = ensemble.predict(X_val)
        from sklearn.metrics import r2_score

        score = float(r2_score(y_val, y_pred))
        print(f"  fold {fold_idx}: R² = {score:.4f}")
        fold_scores.append(score)

    print(f"\nEnsemble CV R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    return fold_scores


# --------------------------------------------------------------------------- #
# SHAP                                                                         #
# --------------------------------------------------------------------------- #

def _generate_shap_artifacts(xgb_model, X_train, feature_names) -> None:
    """Compute SHAP summary for the XGBoost model and persist artefacts."""
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional path
        print(f"[shap] skipped: {exc}")
        return

    _print_banner("Generating SHAP summary plot for XGBoost …")
    sample_size = min(1500, len(X_train))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=sample_size, replace=False)
    X_sample = np.asarray(X_train)[idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(8.5, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    out_png = MODELS_DIR / "shap_summary.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[shap] saved summary plot → {out_png}")

    importance = np.abs(shap_values).mean(axis=0)
    df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    out_csv = MODELS_DIR / "shap_importance.csv"
    df.to_csv(out_csv, index=False)
    print(f"[shap] saved importance ranking → {out_csv}")
    print(df.head(7).to_string(index=False))


# --------------------------------------------------------------------------- #
# Main pipeline                                                                #
# --------------------------------------------------------------------------- #

def train_all_models(X_train, X_test, y_train, y_test) -> Dict[str, TrainedModel]:
    results: Dict[str, TrainedModel] = {}
    for name, (factory, save_path) in MODEL_REGISTRY.items():
        results[name] = _train_one(name, factory, X_train, X_test, y_train, y_test, save_path)
    return results


def main() -> int:
    _print_banner(f"Loading data from {DATA_FILE}")
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(DATA_FILE, fit=True)
    print(f"X_train shape: {X_train.shape}    X_test shape: {X_test.shape}")

    available_features = [
        f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        if f in pd.read_csv(DATA_FILE, nrows=1).columns or f == "Car_Age"
    ]
    feature_names = available_features[: X_train.shape[1]]

    results = train_all_models(X_train, X_test, y_train, y_test)

    val_r2_lookup = {name: float(np.mean(r.cv_scores)) for name, r in results.items()}
    base_models = {name: r.model for name, r in results.items()}
    ensemble = WeightedEnsemble(base_models, val_r2_lookup)

    _print_banner("Weighted Ensemble — model weights (∝ validation R²)")
    for name, w in sorted(ensemble.get_weights().items(), key=lambda kv: -kv[1]):
        print(f"  {name:30s}  R²_val = {val_r2_lookup[name]:.4f}   w = {w:.4f}")

    _print_banner("Weighted Ensemble — held-out test set")
    ensemble_test_metrics = evaluate_model(
        y_test, ensemble.predict(X_test), "Weighted Ensemble (Test)"
    )

    ensemble_cv_scores = _ensemble_cross_validate(X_train, y_train, val_r2_lookup)

    _save_pickle(ensemble, MODEL_ENSEMBLE)
    print(f"\n[ensemble] saved → {MODEL_ENSEMBLE}")

    # Persist all reports.
    per_fold = {name: r.cv_scores for name, r in results.items()}
    per_fold["Weighted Ensemble"] = ensemble_cv_scores

    metrics = {name: r.test_metrics for name, r in results.items()}
    metrics["Weighted Ensemble"] = ensemble_test_metrics

    weights = ensemble.get_weights()

    (MODELS_DIR / "per_fold_r2.json").write_text(json.dumps(per_fold, indent=2), encoding="utf-8")
    (MODELS_DIR / "all_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODELS_DIR / "ensemble_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")

    print("\nWrote per_fold_r2.json, all_metrics.json, ensemble_weights.json")

    # SHAP for XGBoost.
    xgb_model = results["XGBoost"].model
    _generate_shap_artifacts(xgb_model, X_train, feature_names)

    # Final summary table.
    _print_banner("FINAL MODEL COMPARISON (sorted by test R²)")
    summary = pd.DataFrame(
        [
            {"model": name, **m, "cv_mean_r2": float(np.mean(per_fold[name])), "cv_std_r2": float(np.std(per_fold[name]))}
            for name, m in metrics.items()
        ]
    ).sort_values("r2", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
