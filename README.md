# Car Price Prediction — 9-model ensemble

A complete, reproducible Master's-thesis project for predicting used car prices
with a heterogeneous ensemble of nine machine-learning algorithms and an
explicit performance-weighted voting scheme.

> Companion thesis: *Car Price Prediction Using Multi-Model Ensemble Machine
> Learning* (Jasser Abbassi). The numbers reported in the thesis can be
> reproduced from this repository with the four commands in
> [§ Reproducibility](#reproducibility).

---

## Architecture

| # | Model | Library | Role |
|---|-------|---------|------|
| 1 | Linear Regression          | scikit-learn | linear baseline |
| 2 | Ridge Regression           | scikit-learn | L2-regularised baseline |
| 3 | Lasso Regression           | scikit-learn | L1-regularised baseline |
| 4 | Support Vector Regression  | scikit-learn | RBF-kernel non-linear baseline |
| 5 | Random Forest              | scikit-learn | bagging |
| 6 | Gradient Boosting          | scikit-learn | sequential boosting |
| 7 | XGBoost                    | xgboost      | regularised boosting |
| 8 | AdaBoost                   | scikit-learn | adaptive boosting |
| 9 | MLP (3 hidden layers)      | scikit-learn | shallow neural network |
| 10 | **Weighted Ensemble**      | this repo    | ŷ = Σ wₖ ŷₖ, wₖ ∝ R²ₖ_validation, Σ wₖ = 1 |

The weighted-ensemble class lives in [`train_model.py`](train_model.py).
Negative validation R² scores are clipped to zero so a poor base model cannot
push the ensemble in the wrong direction.

---

## Feature engineering

`utils.add_engineered_features()` derives **`Car_Age = CURRENT_YEAR − Year`**
(see [`config.py`](config.py)) before outlier removal, so depreciation enters
linear models as a monotonic feature and reduces the bias caused by using the
raw manufacture year.

Final feature set fed to every model:

```
NUMERICAL_FEATURES   = ["Year", "Car_Age", "Engine Size", "Mileage"]
CATEGORICAL_FEATURES = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]
```

---

## Project layout

```
car-price-prediction/
├── config.py                # paths, hyperparameters, RANDOM_STATE
├── utils.py                 # preprocessing pipeline (Car_Age, scaler, encoder)
├── metrics.py               # R², RMSE, MAE, MAPE, Adj-R²
├── prediction_utils.py      # PredictionHistory + DataQualityReport
├── quality_report.py        # data-quality summary
├── explainability.py        # SHAP / LIME helpers
├── train_model.py           # trains 9 base models + weighted ensemble + SHAP
├── tune_hyperparameters.py  # RandomizedSearchCV (Bergstra & Bengio, 2012)
├── wilcoxon_test.py         # paired Wilcoxon test on per-fold R² values
├── api.py                   # FastAPI service (9 individual + weighted ensemble)
├── app.py                   # Streamlit dashboard
├── tests.py                 # pytest suite (preprocessing, ensemble, metrics)
├── eda.py                   # exploratory data analysis utilities
├── scripts/
│   └── prepare_dataset.py   # downloads + cleans the real Kaggle dataset
├── data/
│   ├── car_price_kaggle_clean.csv  # produced by prepare_dataset.py
│   └── car_price_prediction_.csv   # (legacy synthetic CSV; kept for backward compat)
├── notebooks/               # exploratory Jupyter notebooks
├── models/                  # 9 fitted base models + ensemble + SHAP artefacts (committed)
├── .python-version          # Python 3.11 pin used by Streamlit Community Cloud
├── .streamlit/config.toml   # cohesive blue/slate theme for the dashboard
└── requirements.txt
```

---

## Reproducibility

Recommended Python version: **3.11** (matches the version reported in the thesis).

```bash
# 1. Create environment + install pinned deps
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download + clean the real Kaggle dataset (≈15k rows after filtering)
python scripts/prepare_dataset.py

# 3. (Optional) tune RF/XGB/GB/MLP and overwrite the *_PARAMS dicts in config.py
python tune_hyperparameters.py --update-config

# 4. Train all 9 base models + weighted ensemble (also generates SHAP summary)
python train_model.py

# 5. Wilcoxon signed-rank test (Ensemble vs every base model)
python wilcoxon_test.py
```

After step 4 the `models/` folder contains:

| File | Purpose |
|------|---------|
| `linear_model.pkl … mlp_model.pkl` | 9 fitted base models |
| `weighted_ensemble.pkl`            | fitted `WeightedEnsemble` |
| `scaler.pkl`, `encoder.pkl`        | preprocessing artefacts |
| `all_metrics.json`                 | held-out test R² / RMSE / MAE / MAPE per model |
| `per_fold_r2.json`                 | per-fold R² for all 9 models + ensemble |
| `ensemble_weights.json`            | weights wₖ used by the ensemble |
| `shap_summary.png`                 | SHAP summary plot for XGBoost |
| `shap_importance.csv`              | mean \|SHAP\| ranking |

After step 5 you also get:

| File | Purpose |
|------|---------|
| `wilcoxon_results.md`              | human-readable significance table |
| `wilcoxon_results.json`            | machine-readable W and p-values |

---

## Serving

```bash
# Streamlit dashboard (Home / Predictor / EDA / Model Details / Statistical Tests / …)
streamlit run app.py

# FastAPI REST service
uvicorn api:app --reload --port 8000
```

Sample request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{
    "Brand": "Toyota", "Model": "Corolla", "Year": 2018,
    "Engine Size": 1.8, "Mileage": 65000,
    "Fuel Type": "Petrol", "Transmission": "Automatic", "Condition": "Used"
  }'
```

The API returns the prediction of every base model, the weighted-ensemble
prediction, the ensemble weights, and a 95 % normal-approximation confidence
range derived from the inter-model standard deviation.

---

## One-click cloud deployment (Streamlit Community Cloud)

The repo is deployment-ready: `requirements.txt` pins every dependency,
`.python-version` pins Python 3.11, and the trained model artefacts are
committed under `models/` (≈121 MB) so the app can serve predictions on a
cold-start without retraining.

1. Sign in at <https://share.streamlit.io> with your GitHub account and grant
   Streamlit access to this repo.
2. Click **New app** → pick `jasserabbassi/car-price-prediction` → branch
   `cppv2` → main file `app.py`.
3. Click **Deploy**. The first build runs `pip install -r requirements.txt`
   (≈2–3 min); subsequent pushes to `cppv2` auto-redeploy.

The free tier provides 1 GB RAM, which is sufficient for the 9 fitted models
and the SHAP `TreeExplainer` used on the *Price Predictor* page.

---

## Tests

```bash
pytest tests.py -v
```

Covers data loading, Car_Age engineering, outlier removal, metric correctness,
`PredictionHistory` IO, `WeightedEnsemble` invariants (weights sum to 1,
negative R² clipped to zero), and — once `train_model.py` has run — a
performance gate (best model R² ≥ 0.80) and a CV-stability gate
(weighted-ensemble fold-std ≤ 0.05).

---

## Citations

If you use this code, please cite the thesis and the underlying methods:

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD '16.
- Bergstra, J., & Bengio, Y. (2012). *Random search for hyper-parameter optimization*. JMLR, 13.
- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*. NeurIPS.
- Chai, T., & Draxler, R. R. (2014). *Root mean square error (RMSE) or mean absolute error (MAE)?*. Geosci. Model Dev., 7.

## License

MIT — see [LICENSE](LICENSE).
