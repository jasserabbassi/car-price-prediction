"""End-to-end tests for the 9-model car-price-prediction project.

Run::

    pytest tests.py -v

Tests cover:
    - Data loading + Car_Age engineering
    - Outlier removal + missing-value handling
    - Metric correctness (perfect / non-trivial cases)
    - PredictionHistory IO
    - WeightedEnsemble class invariants (weights sum to 1, predict shape)
    - Trained-model artefacts on disk (skipped if not yet trained)
    - End-to-end performance gate: R² >= MIN_R2 for at least one base model
    - CV stability gate: weighted-ensemble CV std <= MAX_ENSEMBLE_CV_STD
"""

from __future__ import annotations

import json
import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CV_SPLITS,
    DATA_FILE,
    MODEL_ENSEMBLE,
    MODEL_RANDOM_FOREST,
    MODEL_XGBOOST,
    MODEL_GRADIENT_BOOSTING,
    MODEL_LINEAR,
    MODEL_RIDGE,
    MODEL_LASSO,
    MODEL_SVR,
    MODEL_ADABOOST,
    MODEL_MLP,
    MODELS_DIR,
)
from metrics import calculate_metrics
from prediction_utils import PredictionHistory
from utils import (
    add_engineered_features,
    handle_missing_values,
    load_data,
    remove_outliers,
)

# Quality gates (looser than the thesis target so the suite is robust to seeds
# and to the natural ceiling of the real Kaggle dataset).
MIN_R2 = 0.65
MAX_ENSEMBLE_CV_STD = 0.05


# --------------------------------------------------------------------------- #
# Data loading & preprocessing                                                 #
# --------------------------------------------------------------------------- #


class TestDataLoading(unittest.TestCase):
    def test_load_data_returns_dataframe(self) -> None:
        df = load_data(DATA_FILE)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 100)
        self.assertIn("Brand", df.columns)
        self.assertIn("Price", df.columns)

    def test_missing_value_handling(self) -> None:
        df = pd.DataFrame(
            {
                "Brand": ["Toyota", None, "BMW"],
                "Year": [2020, 2021, np.nan],
                "Price": [25000, 30000, 35000],
            }
        )
        clean = handle_missing_values(df)
        self.assertEqual(clean["Brand"].isna().sum(), 0)
        self.assertEqual(clean["Year"].isna().sum(), 0)

    def test_car_age_engineering(self) -> None:
        df = pd.DataFrame({"Year": [2000, 2010, 2020]})
        engineered = add_engineered_features(df, current_year=2024)
        self.assertIn("Car_Age", engineered.columns)
        np.testing.assert_array_equal(engineered["Car_Age"].values, [24.0, 14.0, 4.0])

    def test_outlier_removal(self) -> None:
        df = pd.DataFrame(
            {
                "Year": np.r_[np.full(99, 2018), [1000]],  # 1000 is an outlier
                "Mileage": np.r_[np.full(99, 50_000), [50_000]],
                "Engine Size": np.r_[np.full(99, 2.0), [2.0]],
                "Price": np.r_[np.full(99, 20_000), [20_000]],
            }
        )
        cleaned = remove_outliers(df)
        self.assertNotIn(1000, cleaned["Year"].values)
        self.assertGreaterEqual(len(cleaned), 90)


# --------------------------------------------------------------------------- #
# PredictionHistory                                                            #
# --------------------------------------------------------------------------- #


class TestPredictionHistory(unittest.TestCase):
    TEST_FILE = "test_history_temporary.json"

    def setUp(self) -> None:
        self.history = PredictionHistory(self.TEST_FILE)

    def tearDown(self) -> None:
        Path(self.TEST_FILE).unlink(missing_ok=True)

    def test_add_and_export(self) -> None:
        features = {"Brand": "Toyota", "Year": 2020}
        predictions = {"Linear Regression": 25_000.0, "XGBoost": 26_000.0}
        self.history.add_prediction(features, predictions, list(predictions.keys()))
        self.assertEqual(len(self.history.predictions), 1)
        self.assertIn("timestamp", self.history.predictions[-1])
        self.assertIn("average_prediction", self.history.predictions[-1])


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        m = calculate_metrics(y, y)
        self.assertAlmostEqual(m["MAE"], 0, places=5)
        self.assertAlmostEqual(m["RMSE"], 0, places=5)
        self.assertAlmostEqual(m["R2"], 1.0, places=5)

    def test_consistency(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 38.0])
        m = calculate_metrics(y_true, y_pred)
        self.assertGreater(m["MAE"], 0)
        self.assertGreater(m["RMSE"], 0)


# --------------------------------------------------------------------------- #
# WeightedEnsemble class invariants                                            #
# --------------------------------------------------------------------------- #


class TestWeightedEnsemble(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        from sklearn.dummy import DummyRegressor
        from train_model import WeightedEnsemble

        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 3))
        y = rng.normal(size=(10,))
        models = {f"m{i}": DummyRegressor(strategy="mean").fit(X, y) for i in range(3)}
        ens = WeightedEnsemble(models, {"m0": 0.5, "m1": 0.7, "m2": 0.9})
        self.assertAlmostEqual(sum(ens.get_weights().values()), 1.0, places=6)

    def test_negative_r2_clipped_to_zero(self) -> None:
        from sklearn.dummy import DummyRegressor
        from train_model import WeightedEnsemble

        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 3))
        y = rng.normal(size=(10,))
        models = {f"m{i}": DummyRegressor(strategy="mean").fit(X, y) for i in range(2)}
        ens = WeightedEnsemble(models, {"m0": 0.8, "m1": -0.3})
        weights = ens.get_weights()
        self.assertEqual(weights["m1"], 0.0)
        self.assertAlmostEqual(weights["m0"], 1.0, places=6)


# --------------------------------------------------------------------------- #
# Trained-model artefacts                                                      #
# --------------------------------------------------------------------------- #


class TestTrainedArtefacts(unittest.TestCase):
    """Skipped if models/ has not been populated by train_model.py."""

    REQUIRED_PATHS = [
        MODEL_LINEAR, MODEL_RIDGE, MODEL_LASSO, MODEL_SVR,
        MODEL_RANDOM_FOREST, MODEL_GRADIENT_BOOSTING,
        MODEL_XGBOOST, MODEL_ADABOOST, MODEL_MLP, MODEL_ENSEMBLE,
    ]

    def setUp(self) -> None:
        if not all(Path(p).exists() for p in self.REQUIRED_PATHS):
            self.skipTest("Models not yet trained — run `python train_model.py` first.")

    def test_all_models_loadable(self) -> None:
        for path in self.REQUIRED_PATHS:
            with open(path, "rb") as f:
                model = pickle.load(f)
            self.assertIsNotNone(model)

    def test_metrics_json_present(self) -> None:
        path = MODELS_DIR / "all_metrics.json"
        self.assertTrue(path.exists(), f"{path} missing — train_model.py did not finish")
        metrics = json.loads(path.read_text())
        self.assertIn("Weighted Ensemble", metrics)
        self.assertIn("XGBoost", metrics)

    def test_at_least_one_base_model_meets_r2(self) -> None:
        path = MODELS_DIR / "all_metrics.json"
        if not path.exists():
            self.skipTest("metrics json not present")
        metrics = json.loads(path.read_text())
        max_r2 = max(m["r2"] for m in metrics.values())
        self.assertGreaterEqual(
            max_r2, MIN_R2,
            f"No model achieved R² ≥ {MIN_R2} (max was {max_r2:.4f})."
        )

    def test_ensemble_cv_stability(self) -> None:
        path = MODELS_DIR / "per_fold_r2.json"
        if not path.exists():
            self.skipTest("per-fold json not present")
        data = json.loads(path.read_text())
        self.assertIn("Weighted Ensemble", data)
        scores = np.asarray(data["Weighted Ensemble"], dtype=float)
        self.assertEqual(len(scores), CV_SPLITS)
        self.assertLessEqual(
            float(scores.std()), MAX_ENSEMBLE_CV_STD,
            f"Ensemble CV std too large: {scores.std():.4f}",
        )


if __name__ == "__main__":
    unittest.main()
