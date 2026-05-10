"""Utility functions for data preprocessing and model evaluation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
from config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET,
    TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE,
    SCALER_FILE, ENCODER_FILE
)


def load_data(filepath):
    """Load and return the dataset."""
    df = pd.read_csv(filepath)
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # For numerical features, fill with median
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical features, fill with mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return df


def remove_outliers(df, columns=None, threshold=3):
    """Remove outliers using z-score method."""
    if columns is None:
        columns = NUMERICAL_FEATURES + [TARGET]
    
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean


def encode_categorical_features(df, fit=True, encoder=None):
    """Encode categorical features using LabelEncoder."""
    df_encoded = df.copy()
    
    if fit:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
        
        # Save encoders
        os.makedirs(os.path.dirname(ENCODER_FILE), exist_ok=True)
        with open(ENCODER_FILE, 'wb') as f:
            pickle.dump(encoders, f)
        
        return df_encoded, encoders
    else:
        if encoder is None:
            with open(ENCODER_FILE, 'rb') as f:
                encoder = pickle.load(f)
        
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                df_encoded[col] = encoder[col].transform(df_encoded[col].astype(str))
        
        return df_encoded, encoder


def scale_features(X_train, X_test, fit=True, scaler=None):
    """Scale numerical features using StandardScaler."""
    if fit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        
        return X_train_scaled, X_test_scaled, scaler
    else:
        if scaler is None:
            with open(SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler


def preprocess_data(filepath, fit=True, encoder=None, scaler=None):
    """Complete preprocessing pipeline."""
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove outliers
    df = remove_outliers(df)
    
    # Encode categorical features
    df, encoder = encode_categorical_features(df, fit=fit, encoder=encoder)
    
    # Separate features and target, then select in EXACT order
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[all_features].astype(float)
    y = df[TARGET]
    
    # Split data
    if fit:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Scale features
        X_train, X_test, scaler = scale_features(X_train, X_test, fit=True)
        
        return X_train, X_test, y_train, y_test, scaler, encoder
    else:
        if scaler is None:
            with open(SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
        X_scaled, _, scaler = scale_features(X, X, fit=False, scaler=scaler)
        return X_scaled, scaler, encoder


def preprocess_input_data(input_dict, encoder, scaler):
    """Preprocess user input data for prediction."""
    try:
        # Define feature order to match training data
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        
        # Convert to DataFrame
        df = pd.DataFrame([input_dict])
        
        # Ensure all required features are present
        for feature in all_features:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Validate categorical features against encoder classes
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col in encoder:
                value = str(df[col].iloc[0])
                valid_classes = list(encoder[col].classes_)
                if value not in valid_classes:
                    raise ValueError(
                        f"Invalid value '{value}' for {col}. Valid options: {valid_classes}"
                    )
        
        # Encode categorical features
        df_encoded = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns and col in encoder:
                df_encoded[col] = encoder[col].transform(df_encoded[col].astype(str))
        
        # Select features in EXACT order used during training
        X = df_encoded[all_features].astype(float)
        
        # Validate feature shape
        if X.shape[1] != len(all_features):
            raise ValueError(f"Feature shape mismatch: expected {len(all_features)}, got {X.shape[1]}")
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluate model performance with multiple metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mse': mse
    }


def get_feature_importance(model, feature_names):
    """Extract feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importance_dict = {}
        for i in range(len(feature_names)):
            feature_importance_dict[feature_names[i]] = importances[i]
        
        return feature_importance_dict
    return None
