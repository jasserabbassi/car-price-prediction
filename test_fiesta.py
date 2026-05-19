"""Predict the Ford Fiesta test case with current models."""
import joblib
import numpy as np
import pandas as pd
from config import (
    CATEGORICAL_FEATURES, CURRENT_YEAR, ENCODER_FILE, MODEL_ADABOOST,
    MODEL_GRADIENT_BOOSTING, MODEL_LINEAR, MODEL_LASSO, MODEL_MLP,
    MODEL_RANDOM_FOREST, MODEL_RIDGE, MODEL_SVR, MODEL_XGBOOST,
    NUMERICAL_FEATURES, SCALER_FILE,
)
import json

# Load preprocessing artifacts
scaler = joblib.load(SCALER_FILE)
encoder = joblib.load(ENCODER_FILE)

# Load all models
model_paths = {
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

models = {}
for name, path in model_paths.items():
    try:
        models[name] = joblib.load(path)
        print(f"Loaded {name}")
    except Exception as e:
        print(f"Failed to load {name}: {e}")

# Load ensemble weights
weights_path = "models/ensemble_weights.json"
with open(weights_path, "r") as f:
    weights = json.load(f)

# Ford Fiesta test case
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

print(f"\nInput: Ford Fiesta, 2015, 90,000 mi, 1.2L, Petrol, Automatic, Used")
print(f"Car_Age = {CURRENT_YEAR} - 2015 = {CURRENT_YEAR - 2015}")

# Preprocess
df = pd.DataFrame([test_input])
df["Car_Age"] = float(CURRENT_YEAR) - float(df["Year"].iloc[0])

for col in CATEGORICAL_FEATURES:
    if col in encoder:
        df[col] = encoder[col].transform(df[col].astype(str))

all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
X = df[all_features].astype(float)
X_scaled = scaler.transform(X)

print(f"\nFeature vector (scaled): {X_scaled[0]}")
print(f"Feature names: {all_features}")

# Predict with each model
print("\n=== Individual Model Predictions ===")
predictions = {}
for name, model in models.items():
    try:
        pred = float(model.predict(X_scaled)[0])
        pred = max(0, pred)
        predictions[name] = pred
        w = weights.get(name, 0)
        print(f"  {name:30s}  ${pred:>10,.0f}  (weight: {w:.4f})")
    except Exception as e:
        print(f"  {name:30s}  Error: {e}")

# Ensemble prediction
from train_model import WeightedEnsemble
val_r2 = {name: 0.5 for name in weights.keys()}
ensemble = WeightedEnsemble(models, val_r2, top_k=None)
ensemble.weights = {k: float(v) for k, v in weights.items()}
ensemble.models = {k: models[k] for k in ensemble.weights.keys()}

ensemble_pred = float(ensemble.predict(X_scaled)[0])
ensemble_pred = max(0, ensemble_pred)
print(f"\n  {'Weighted Ensemble':30s}  ${ensemble_pred:>10,.0f}")

# Statistics
preds = list(predictions.values())
print(f"\n=== Summary ===")
print(f"  Average:  ${np.mean(preds):>10,.0f}")
print(f"  Std Dev:  ${np.std(preds):>10,.0f}")
print(f"  Min:      ${np.min(preds):>10,.0f}")
print(f"  Max:      ${np.max(preds):>10,.0f}")
print(f"  Range:    ${np.max(preds) - np.min(preds):>10,.0f}")
print(f"\nExpected: ~$5,000-$8,000")
print(f"Ensemble: ${ensemble_pred:,.0f}")
if ensemble_pred > 10000:
    print(f"OVERESTIMATED by ${ensemble_pred - 8000:,.0f} (vs upper bound)")
elif ensemble_pred < 5000:
    print(f"UNDERESTIMATED by ${5000 - ensemble_pred:,.0f} (vs lower bound)")
else:
    print(f"WITHIN expected range!")
