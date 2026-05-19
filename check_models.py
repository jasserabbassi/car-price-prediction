"""Check feature importances and model metrics."""
import pickle
import json
import numpy as np
from config import *

weights_path = MODELS_DIR / 'ensemble_weights.json'
if weights_path.exists():
    with open(weights_path, 'r') as f:
        weights = json.load(f)
    print('=== ENSEMBLE WEIGHTS ===')
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f'  {name:30s}  w = {w:.4f}')

metrics_path = MODELS_DIR / 'all_metrics.json'
if metrics_path.exists():
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    print('\n=== MODEL METRICS ===')
    for name, m in sorted(metrics.items(), key=lambda x: -x[1].get('r2', 0)):
        r2 = m['r2']
        rmse = m['rmse']
        mae = m['mae']
        print(f'  {name:30s}  R2={r2:.4f}  RMSE={rmse:,.0f}  MAE={mae:,.0f}')

print('\n=== FEATURE IMPORTANCE (tree-based models) ===')
feature_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
for name, path in [('Random Forest', MODEL_RANDOM_FOREST), ('XGBoost', MODEL_XGBOOST), ('Gradient Boosting', MODEL_GRADIENT_BOOSTING), ('AdaBoost', MODEL_ADABOOST)]:
    if path.exists():
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            print(f'\n  {name}:')
            for fname, val in sorted(zip(feature_names, imp), key=lambda x: -x[1]):
                print(f'    {fname:20s}  {val:.4f}')
