# Advanced Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Advanced Models](#advanced-models)
4. [Model Explainability](#model-explainability)
5. [Deployment Guide](#deployment-guide)
6. [Performance Metrics](#performance-metrics)

---

## Project Overview

**Car Price Prediction System** - An enterprise-grade machine learning application with:
- 9 advanced ML models
- Real-time predictions via Streamlit web interface
- Production API with FastAPI
- Comprehensive EDA and data quality analysis
- Model explainability using SHAP values
- Prediction history and confidence intervals

---

## System Architecture

```
┌─────────────────────────────────────────┐
│     Data Input (CSV/API/Web Form)       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   Data Preprocessing & Validation       │
│  - Missing Value Imputation              │
│  - Outlier Detection & Handling          │
│  - Feature Encoding (Categorical)        │
│  - Feature Scaling (Numerical)           │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Model Training & Validation        │
│  - 9 Different ML Models                 │
│  - Cross-Validation (5-Fold)             │
│  - Hyperparameter Tuning                 │
│  - Model Persistence (Pickle)            │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┬──────────────┐
       │                │              │
┌──────▼────────┐ ┌────▼──────┐ ┌─────▼──────┐
│ Streamlit Web │ │ FastAPI   │ │ Python CLI │
│   Interface   │ │    API    │ │   Scripts  │
└───────────────┘ └───────────┘ └────────────┘
```

---

## Advanced Models

### 1. Linear Regression (Baseline)
- **Type**: Linear regression
- **Use**: Quick baseline comparisons
- **Pros**: Interpretable, fast
- **Cons**: May underfit complex relationships

### 2. Ridge & Lasso Regression
- **Type**: Regularized linear regression
- **Use**: Prevent overfitting
- **Parameters**: 
  - Ridge: α=1.0
  - Lasso: α=1.0

### 3. Support Vector Regression (SVR)
- **Type**: Non-parametric, kernel-based
- **Kernel**: RBF (Radial Basis Function)
- **C**: 100
- **Gamma**: 'scale'

### 4. Random Forest
- **Type**: Ensemble of decision trees
- **Estimators**: 200
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Feature Importance**: Yes

### 5. XGBoost
- **Type**: Gradient boosting
- **Estimators**: 150
- **Learning Rate**: 0.1
- **Max Depth**: 7
- **Subsample**: 0.8
- **Colsample Bytree**: 0.8
- **Feature Importance**: Yes

### 6. Gradient Boosting
- **Type**: Sequential boosting
- **Estimators**: 200
- **Learning Rate**: 0.1
- **Max Depth**: 5
- **Min Samples Split**: 5
- **Subsample**: 0.8

### 7. AdaBoost
- **Type**: Adaptive boosting
- **Base Estimator**: Decision Tree
- **Estimators**: 100
- **Learning Rate**: 1.0

### 8. Ensemble Voting Regressor
- **Type**: Meta-ensemble
- **Voters**: Random Forest + XGBoost + Gradient Boosting
- **Weights**: Equal (1, 1, 1)

### 9. Neural Network (Optional)
- **Type**: Multi-layer perceptron
- **Layers**: [128, 64, 32]
- **Activation**: ReLU
- **Loss**: MSE

---

## Model Explainability

### SHAP (SHapley Additive exPlanations)

**What it does:**
- Explains individual predictions
- Shows feature contributions to price
- Generates force plots and summary plots

**Key Features:**
```python
from explainability import ModelExplainer

# Create explainer
explainer = ModelExplainer(model, X_train, feature_names)

# Explain single prediction
explanation = explainer.explain_prediction(X_instance)

# Get feature importance
importance_df = explainer.get_feature_importance()
```

**Interpretation:**
- Red bars: Increase price
- Blue bars: Decrease price
- Bar length: Magnitude of impact

---

## Performance Metrics

### Regression Metrics

#### R² Score
- Range: 0-1 (higher is better)
- Interpretation: % of variance explained
- Formula: 1 - (SS_res / SS_tot)

#### RMSE (Root Mean Squared Error)
- Unit: Price currency (same as target)
- Penalizes large errors heavily
- Formula: √(Σ(y_true - y_pred)² / n)

#### MAE (Mean Absolute Error)
- Unit: Price currency
- More robust to outliers than RMSE
- Formula: Σ|y_true - y_pred| / n

#### MAPE (Mean Absolute Percentage Error)
- Unit: Percentage %
- Normalized metric for comparison
- Formula: (100/n) * Σ(|y_true - y_pred| / y_true)

### Model Comparison Example
```
Model                 R²        RMSE      MAE       MAPE
─────────────────────────────────────────────────────
Random Forest       0.892     3,450      2,100     8.5%
XGBoost             0.905     3,200      1,950     7.8%
Gradient Boosting   0.898     3,380      2,050     8.2%
SVR                 0.875     3,890      2,450     9.8%
Linear Reg          0.745     5,230      3,100     12.4%
```

---

## Deployment Guide

### Development Setup
```bash
# Clone repository
cd car-price-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models
python train_model.py

# Run Streamlit app
streamlit run app.py

# Run API server
uvicorn api:app --reload
```

### Production Deployment

#### Option 1: AWS EC2 + Docker
```bash
# Build Docker image
docker build -t car-prediction:latest .

# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag car-prediction:latest <account>.dkr.ecr.us-east-1.amazonaws.com/car-prediction:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/car-prediction:latest

# Deploy on EC2
docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/car-prediction:latest
docker run -d -p 8000:8000 car-prediction:latest
```

#### Option 2: Heroku
```bash
# Login
heroku login

# Create app
heroku create car-price-prediction-app

# Add Procfile
echo "web: gunicorn -w 4 -b 0.0.0.0:\$PORT api:app" > Procfile

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

#### Option 3: Google Cloud Run
```bash
# Build and deploy
gcloud run deploy car-price-prediction \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Data Quality Metrics

### Missing Values Handling
- **Strategy**: Forward fill for time series, mean imputation for others
- **Threshold**: Drop columns with >30% missing
- **Variables Handled**: Mileage, Year, Engine Size

### Outlier Detection
- **Method**: IQR (Interquartile Range)
- **Threshold**: 1.5 × IQR
- **Action**: Flag for review or capping

### Feature Engineering
```python
Features Created:
- Age = Current_Year - Car_Year
- Price_Per_Km = Price / Mileage
- Engine_Category = Binned Engine Size
- Brand_Popularity = Brand Frequency
```

---

## Prediction Confidence

### Confidence Intervals
- **Method**: Multiple model predictions (ensemble)
- **Calculation**: Mean ± 1.96 × (StdDev / √n)
- **Confidence Level**: 95%

Example:
```
Point Estimate: $25,560
95% CI: [$25,200 - $25,920]
Margin of Error: ±$360
```

---

## Running Tests

```bash
# Run all tests
pytest tests.py -v

# Run specific test
pytest tests.py::TestPredictionHistory::test_add_prediction -v

# Generate coverage report
pytest tests.py --cov=. --cov-report=html
```

---

## Monitoring & Logging

### Key Metrics to Monitor
- Prediction latency (target: < 200ms)
- Model accuracy drift
- API error rate (target: < 1%)
- Data freshness

### Log Locations
- API logs: `api.log`
- Training logs: `training.log`
- Prediction history: `prediction_history.json`

---

## Contact & Support

For issues or questions:
- GitHub: [repository-url]
- Email: support@example.com
- Documentation: [docs-url]
