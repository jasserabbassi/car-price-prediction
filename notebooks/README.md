# Car Price Prediction - Jupyter Notebooks Guide

This folder contains comprehensive Jupyter notebooks for the car price prediction project. These notebooks are designed to be used both for **learning** and **PFE presentation**.

## 📚 Notebook Overview

### 1. **01_EDA.ipynb** - Exploratory Data Analysis
**Purpose**: Comprehensive exploration of the dataset

**Topics Covered**:
- Dataset overview and basic statistics
- Price distribution analysis
- Numerical features correlation with price
- Categorical features analysis
- Outlier detection using IQR method
- Heatmap correlations
- Key insights summarization

**Best For**: Understanding data structure and relationships

**Key Insights**:
- Price range and distribution
- Most important features
- Data quality metrics
- Outlier statistics

---

### 2. **02_Model_Training.ipynb** - Model Training & Comparison
**Purpose**: Training and comparing 5 machine learning models

**Models Included**:
- Linear Regression (baseline)
- Random Forest
- XGBoost
- Gradient Boosting
- AdaBoost

**Topics Covered**:
- Data preparation and preprocessing
- Feature scaling and encoding
- Train-test split (80-20)
- Model training
- Model evaluation (RMSE, MAE, R², CV)
- Performance comparison
- Feature importance analysis
- Actual vs Predicted visualization
- Residual analysis

**Best For**: Understanding model training and performance

**Key Metrics**:
- Model accuracy comparison
- Feature importance ranking
- Cross-validation scores
- Prediction error analysis

---

### 3. **03_Model_Predictions.ipynb** - Predictions & Analysis
**Purpose**: Making predictions and comparing model outputs

**Topics Covered**:
- Loading trained models and preprocessors
- Sample car predictions
- Multi-model prediction comparison
- Prediction statistics (mean, std, range)
- Feature importance from different models
- Ensemble predictions (average voting)
- Individual vs ensemble comparison

**Best For**: Demonstrating model predictions and ensemble voting

**Key Features**:
- Side-by-side model predictions
- Prediction uncertainty analysis
- Feature contribution visualization
- Ensemble voting demonstration

---

### 4. **04_Complete_Guide.ipynb** - Complete Project Guide
**Purpose**: Executive summary of the entire project (START HERE!)

**Sections**:
- Project overview and objectives
- Technology stack explanation
- Dataset information
- Model performance summary
- Streamlit dashboard features
- FastAPI REST endpoints
- Running instructions for all components
- Deployment options
- Key achievements for PFE
- Future enhancements

**Best For**: PFE presentation and project overview

**Why Start Here?**: Provides context for the entire project before diving into individual notebooks

---

## 🚀 How to Use These Notebooks

### Prerequisites
```bash
# Install Jupyter
pip install jupyter notebook

# Or use JupyterLab
pip install jupyterlab
```

### Running Notebooks

#### Option 1: Jupyter Notebook (Classic)
```bash
cd notebooks
jupyter notebook
```
Then click on any `.ipynb` file to open it.

#### Option 2: JupyterLab (Recommended)
```bash
cd notebooks
jupyter lab
```

#### Option 3: VS Code (Built-in)
1. Open VS Code
2. Install "Jupyter" extension by Microsoft
3. Open any `.ipynb` file
4. Click "Select Kernel" and choose Python 3.11

---

## 📖 Recommended Reading Order

### For Learning:
1. **04_Complete_Guide.ipynb** - Understand the project
2. **01_EDA.ipynb** - Explore the data
3. **02_Model_Training.ipynb** - Learn model training
4. **03_Model_Predictions.ipynb** - See predictions

### For PFE Presentation:
1. **04_Complete_Guide.ipynb** - Start with overview (5 minutes)
2. **01_EDA.ipynb** - Show data exploration (5 minutes)
3. **02_Model_Training.ipynb** - Demonstrate model comparison (10 minutes)
4. **03_Model_Predictions.ipynb** - Show predictions (5 minutes)
5. **Live Demo**: Switch to Streamlit web app

---

## 🔧 Required Data & Models

The notebooks expect the following structure:
```
tryharding/
├── data/
│   └── car_price_prediction_.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── adaboost_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
└── notebooks/
    ├── 01_EDA.ipynb
    ├── 02_Model_Training.ipynb
    ├── 03_Model_Predictions.ipynb
    └── 04_Complete_Guide.ipynb
```

### Training Models
If models don't exist, train them first:
```bash
cd ..
python train_model.py
cd notebooks
```

---

## 📊 Key Visualization Features

Each notebook includes multiple visualization types:

### 01_EDA.ipynb
- Histograms and box plots
- Scatter plots
- Bar charts by category
- Heatmap correlations
- Outlier visualizations

### 02_Model_Training.ipynb
- Model performance comparison
- Feature importance charts
- Actual vs Predicted plots
- Residual analysis

### 03_Model_Predictions.ipynb
- Multi-model prediction comparison
- Feature importance rankings
- Ensemble voting visualization

---

## 🎯 Use Cases

### For Learning
- Understand ML pipeline from data to production
- Learn different model types and their trade-offs
- Practice data preprocessing and feature scaling
- Understand model evaluation metrics

### For PFE Presentation
- Demonstrate comprehensive project scope
- Show data-driven decision making
- Present model comparison results
- Explain feature importance
- Showcase ensemble voting approach

### For Business Analysis
- Understand car pricing factors
- Explore market trends by brand
- Analyze price distributions
- Identify pricing patterns

---

## 💡 Key Learnings

### Data Processing
- Handling missing values
- Outlier detection and removal
- Categorical encoding
- Feature scaling

### Machine Learning
- Cross-validation for model robustness
- Hyperparameter effects
- Ensemble methods
- Feature importance ranking

### Model Evaluation
- Multiple evaluation metrics (R², RMSE, MAE, MAPE)
- Train vs test performance
- Generalization ability
- Prediction confidence

---

## 🔗 Integration with Web App

**Corresponding Streamlit Pages**:
- **01_EDA.ipynb** → Streamlit "Data Analysis" page
- **02_Model_Training.ipynb** → Streamlit "Model Details" page
- **03_Model_Predictions.ipynb** → Streamlit "Price Predictor" page
- **04_Complete_Guide.ipynb** → Streamlit "Home" page

**Live Demo**: After running notebooks, launch the Streamlit app:
```bash
cd ..
streamlit run app.py
```

---

## 📝 Execution Tips

1. **Run cells sequentially** - Don't skip cells, some set up variables for later ones
2. **Kernel restart**: If you get variable errors, restart kernel (Kernel → Restart)
3. **Long operations**: Model training cells may take 1-2 minutes
4. **Data paths**: Notebooks assume you're running from the `notebooks/` folder
5. **Dependencies**: Ensure all imports work by running first cell

---

## ✅ Validation

To verify everything is working:

```python
# Run this in any notebook cell
import pandas as pd
print("✓ Pandas imported")

import plotly
print("✓ Plotly imported")

import sklearn
print("✓ Scikit-learn imported")

# Check data exists
df = pd.read_csv('../data/car_price_prediction_.csv')
print(f"✓ Data loaded: {df.shape[0]} rows")
```

If all prints show ✓, you're ready to go!

---

## 🎓 Learning Outcomes

After going through these notebooks, you'll understand:

✅ Full machine learning pipeline  
✅ Data exploration and visualization  
✅ Model selection and comparison  
✅ Prediction and ensemble methods  
✅ Performance evaluation metrics  
✅ Feature importance analysis  
✅ Real-world ML applications  

---

## 📞 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'xxx'`
```bash
pip install -r ../requirements.txt
```

**Issue**: `FileNotFoundError: '../data/car_price_prediction_.csv'`
- Ensure notebook is run from `notebooks/` folder
- Or update path to `../../data/...` if running from different location

**Issue**: Models not found
```bash
cd ..
python train_model.py
cd notebooks
```

**Issue**: Slow execution
- Neural network training in `02_Model_Training.ipynb` is normal (takes 30-60 seconds)
- Other notebooks should run quickly

---

## 🚀 Next Steps

After exploring these notebooks:

1. **Run the Streamlit app** for interactive predictions
2. **Review API documentation** for integration
3. **Explore source code** (app.py, config.py, etc.)
4. **Prepare your PFE presentation** using notebook content
5. **Deploy to cloud** for production use

---

**Happy Learning! 📚🤖**

For questions or issues, refer to the main project [README.md](../README.md) or [API_DOCUMENTATION.md](../API_DOCUMENTATION.md)
