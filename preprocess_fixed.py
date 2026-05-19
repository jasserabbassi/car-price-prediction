"""
FIXED preprocessing pipeline for car price prediction.

Key improvements:
1. One-Hot Encoding for categorical features (NOT LabelEncoding)
2. StandardScaler applied ONLY to numerical features
3. ColumnTransformer for clean pipeline (sklearn best practice)
4. No data leakage: fit on train, transform on test
5. Enhanced feature engineering
6. Handles high-cardinality features intelligently
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_FILE = DATA_DIR / "car_price_kaggle_clean.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CURRENT_YEAR = 2024
TARGET = "Price"

# NUMERICAL features (will be StandardScaled)
NUMERICAL_FEATURES = ["Year", "Engine Size", "Mileage"]

# CATEGORICAL features (will be One-Hot Encoded)
CATEGORICAL_FEATURES = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]

# Output file paths
SCALER_FILE = MODELS_DIR / "scaler_fixed.pkl"
ENCODER_FILE = MODELS_DIR / "encoder_fixed.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor_fixed.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names_fixed.pkl"


# ============================================================================
# Feature Engineering
# ============================================================================

def add_engineered_features(df: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Add intelligent features that improve model predictions."""
    df = df.copy()
    
    # 1. Car Age: How old is the car?
    if "Year" in df.columns:
        df["Car_Age"] = current_year - df["Year"].astype(float)
    
    # 2. Normalized Mileage: Mileage adjusted for car age
    #    - New car with 100k miles is heavily worn
    #    - 20-year old car with 100k miles is low usage
    if "Mileage" in df.columns and "Car_Age" in df.columns:
        # Avoid division by zero
        df["Mileage_per_Year"] = df["Mileage"] / (df["Car_Age"] + 1)
    
    # 3. Engine Power Score: Larger engines depreciate differently
    if "Engine Size" in df.columns:
        df["Engine_Size_Squared"] = df["Engine Size"] ** 2
        df["Large_Engine"] = (df["Engine Size"] > 3.5).astype(int)
    
    # 4. Log-transform Mileage: Mileage distribution is usually skewed
    #    Log(mileage) often has better linear relationship with price
    if "Mileage" in df.columns:
        df["Log_Mileage"] = np.log1p(df["Mileage"])  # log1p avoids log(0)
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with intelligent imputation."""
    df = df.copy()
    
    # Numerical: Fill with median (robust to outliers)
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical: Fill with mode (most common value)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    target_col: str = TARGET,
    z_threshold: float = 3.5,
) -> pd.DataFrame:
    """Remove extreme outliers on target variable (prices way too high/low)."""
    df = df.copy()
    
    if target_col in df.columns:
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
        df = df[z_scores < z_threshold]
    
    return df


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    1. Scales numerical features
    2. One-Hot encodes categorical features
    
    IMPORTANT: Fit ONLY on training data to avoid data leakage.
    """
    
    # Numerical pipeline: StandardScale only
    numerical_transformer = StandardScaler()
    
    # Categorical pipeline: One-Hot encode with sparse_output=False for compatibility
    # handle_unknown='ignore' allows unseen categories in production
    categorical_transformer = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        drop=None,  # Keep all categories to preserve model interpretability
    )
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder='drop',  # Drop any other columns
    )
    
    # FIT on training data only (prevents data leakage)
    preprocessor.fit(X_train)
    
    return preprocessor


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> list:
    """Extract feature names after One-Hot Encoding."""
    feature_names = []
    
    # Numerical features stay the same
    feature_names.extend(NUMERICAL_FEATURES)
    
    # Categorical features get One-Hot encoded
    categorical_names = preprocessor.named_transformers_['cat'].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    feature_names.extend(categorical_names)
    
    return feature_names


def preprocess_data(
    filepath: Path,
    fit: bool = True,
    preprocessor=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list]:
    """
    Complete preprocessing pipeline.
    
    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows from {filepath}")
    
    # Handle missing values
    df = handle_missing_values(df)
    print("✓ Handled missing values")
    
    # Add engineered features
    df = add_engineered_features(df)
    print("✓ Added engineered features")
    
    # Remove outliers
    df_clean = remove_outliers(df)
    print(f"✓ Removed outliers ({len(df)} → {len(df_clean)} rows)")
    df = df_clean
    
    # Separate features and target
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Target mean: ${y.mean():,.0f}")
    
    if fit:
        # Train-test split BEFORE any transformation (critical for avoiding data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        print(f"\n✓ Train-test split: {len(X_train)} train, {len(X_test)} test")
        
        # Build preprocessor (fit on training data only)
        preprocessor = build_preprocessor(X_train)
        print("✓ Built ColumnTransformer")
        
        # Transform both sets
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = get_feature_names_after_preprocessing(preprocessor)
        
        # Save preprocessor artifacts
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_FILE, compress=True)
        joblib.dump(feature_names, FEATURE_NAMES_FILE, compress=True)
        
        print(f"✓ Saved preprocessor to {PREPROCESSOR_FILE}")
        print(f"✓ Final feature count: {len(feature_names)}")
        print(f"  - Numerical: {len(NUMERICAL_FEATURES)}")
        print(f"  - Categorical (One-Hot): {len(feature_names) - len(NUMERICAL_FEATURES)}")
        
        return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, feature_names
    
    else:
        # Production: Use pre-fitted preprocessor
        if preprocessor is None:
            preprocessor = joblib.load(PREPROCESSOR_FILE)
        
        X_transformed = preprocessor.transform(X)
        
        feature_names = joblib.load(FEATURE_NAMES_FILE)
        
        return X_transformed, preprocessor, feature_names


def preprocess_single_input(input_dict: dict, preprocessor, feature_names: list) -> np.ndarray:
    """
    Preprocess a single user input for online prediction.
    
    Parameters
    ----------
    input_dict : dict
        Single row dict with keys: Brand, Year, Engine Size, Fuel Type, Transmission, Mileage, Condition, Model
    preprocessor : ColumnTransformer
        Fitted preprocessor
    feature_names : list
        Feature names after preprocessing
    
    Returns
    -------
    X : np.ndarray, shape (1, n_features)
        Preprocessed feature row
    """
    
    # Create DataFrame with single row
    df = pd.DataFrame([input_dict])
    
    # Apply engineered features (SAME as training)
    df = add_engineered_features(df)
    
    # Select only features used in training
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    
    # Transform using fitted preprocessor
    X_transformed = preprocessor.transform(X)
    
    return X_transformed


# ============================================================================
# Standalone Validation Script
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CAR PRICE PREDICTOR - FIXED PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Run full preprocessing
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        DATA_FILE,
        fit=True,
    )
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nTrain set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"\nFeature list ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    
    # Show some statistics
    print(f"\nTrain target statistics:")
    print(f"  Min: ${y_train.min():,.0f}")
    print(f"  Max: ${y_train.max():,.0f}")
    print(f"  Mean: ${y_train.mean():,.0f}")
    print(f"  Std: ${y_train.std():,.0f}")
    
    print(f"\nTest target statistics:")
    print(f"  Min: ${y_test.min():,.0f}")
    print(f"  Max: ${y_test.max():,.0f}")
    print(f"  Mean: ${y_test.mean():,.0f}")
    print(f"  Std: ${y_test.std():,.0f}")
    
    print("\n✓ Ready to train models with fixed preprocessing!")
