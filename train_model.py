"""Train advanced machine learning models for car price prediction."""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, MODELS_DIR, 
    MODEL_RANDOM_FOREST, MODEL_XGBOOST, 
    MODEL_GRADIENT_BOOSTING, MODEL_NEURAL_NETWORK,
    RF_PARAMS, XGB_PARAMS, GB_PARAMS,
    RANDOM_STATE
)
from utils import preprocess_data, evaluate_model, get_feature_importance


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model."""
    print("\n" + "="*50)
    print("Training Random Forest Model...")
    print("="*50)
    
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluation
    print("\nTraining Set Performance:")
    evaluate_model(y_train, y_pred_train, "Random Forest (Train)")
    
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, "Random Forest (Test)")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    print(f"\n5-Fold CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_RANDOM_FOREST, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {MODEL_RANDOM_FOREST}")
    
    return model, test_metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model."""
    print("\n" + "="*50)
    print("Training XGBoost Model...")
    print("="*50)
    
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluation
    print("\nTraining Set Performance:")
    evaluate_model(y_train, y_pred_train, "XGBoost (Train)")
    
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, "XGBoost (Test)")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    print(f"\n5-Fold CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_XGBOOST, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {MODEL_XGBOOST}")
    
    return model, test_metrics


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting model."""
    print("\n" + "="*50)
    print("Training Gradient Boosting Model...")
    print("="*50)
    
    model = GradientBoostingRegressor(**GB_PARAMS)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluation
    print("\nTraining Set Performance:")
    evaluate_model(y_train, y_pred_train, "Gradient Boosting (Train)")
    
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, "Gradient Boosting (Test)")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    print(f"\n5-Fold CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_GRADIENT_BOOSTING, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {MODEL_GRADIENT_BOOSTING}")
    
    return model, test_metrics


def train_adaboost_model(X_train, X_test, y_train, y_test):
    """Train AdaBoost Regression model."""
    print("\n" + "="*50)
    print("Training AdaBoost Regression Model...")
    print("="*50)
    
    model = AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluation
    print("\nTraining Set Performance:")
    evaluate_model(y_train, y_pred_train, "AdaBoost (Train)")
    
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, "AdaBoost (Test)")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    print(f"\n5-Fold CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model, test_metrics


def train_neural_network(X_train, X_test, y_train, y_test, scaler):
    """Train a neural network for price prediction."""
    print("\n" + "="*50)
    print("Training Neural Network Model...")
    print("="*50)
    
    # Build neural network
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        layers.Dense(16, activation='relu'),
        
        layers.Dense(1)  # Output layer
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    print("\nTraining neural network (this may take a moment)...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    
    # Evaluation
    print("\nTraining Set Performance:")
    evaluate_model(y_train, y_pred_train, "Neural Network (Train)")
    
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, "Neural Network (Test)")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(str(MODEL_NEURAL_NETWORK))
    print(f"\nModel saved: {MODEL_NEURAL_NETWORK}")
    
    return model, test_metrics


def train_ensemble_voting_model(X_train, X_test, y_train, y_test, rf_model, xgb_model, gb_model):
    """Train Ensemble Voting Regressor combining best sklearn models."""
    print("\n" + "="*50)
    print("Creating Ensemble Voting Regressor...")
    print("="*50)
    
    # Create voting ensemble with pure sklearn models (RF + GB for stability)
    # Note: XGBoost excluded due to sklearn compatibility issues
    voting_model = VotingRegressor(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        n_jobs=-1
    )
    
    # Fit the ensemble 
    voting_model.fit(X_train, y_train)
    
    # Evaluate the voting ensemble on test data
    y_pred_train = voting_model.predict(X_train)
    y_pred_test = voting_model.predict(X_test)
    
    # Evaluation
    print("\nTraining Set Performance:")
    evaluate_model(y_train, y_pred_train, "Ensemble Voting (Train)")
    
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, "Ensemble Voting (Test)")
    
    return voting_model, test_metrics



def train_all_models(X_train, X_test, y_train, y_test, scaler=None):
    """Train all models and return results."""
    models = {}
    metrics = {}
    
    # Train ensemble models (the most important ones)
    print("\n" + "="*70)
    print("TRAINING ENSEMBLE MODELS")
    print("="*70)
    models['Random Forest'], metrics['Random Forest'] = \
        train_random_forest(X_train, X_test, y_train, y_test)
    
    models['XGBoost'], metrics['XGBoost'] = \
        train_xgboost(X_train, X_test, y_train, y_test)
    
    models['Gradient Boosting'], metrics['Gradient Boosting'] = \
        train_gradient_boosting(X_train, X_test, y_train, y_test)
    
    models['AdaBoost'], metrics['AdaBoost'] = \
        train_adaboost_model(X_train, X_test, y_train, y_test)
    
    # Train neural network
    print("\n" + "="*70)
    print("TRAINING DEEP LEARNING MODEL")
    print("="*70)
    models['Neural Network'], metrics['Neural Network'] = \
        train_neural_network(X_train, X_test, y_train, y_test, scaler)
    
    # Create voting ensemble
    models['Ensemble Voting'], metrics['Ensemble Voting'] = \
        train_ensemble_voting_model(X_train, X_test, y_train, y_test, 
                                    models['Random Forest'], 
                                    models['XGBoost'], 
                                    models['Gradient Boosting'])
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(metrics).T
    results_df = results_df.sort_values('r2', ascending=False)
    print(results_df)
    
    print(f"\n🏆 Best Model: {results_df.index[0]} (R² = {results_df.iloc[0]['r2']:.4f})")
    
    return models, metrics, results_df


def main():
    """Main training pipeline."""
    print("🚀 Starting car price prediction model training...")
    print("="*70)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(DATA_FILE, fit=True)
    
    print(f"✓ Training set size: {X_train.shape}")
    print(f"✓ Test set size: {X_test.shape}")
    print(f"✓ Number of features: {X_train.shape[1]}")
    
    # Train all models (pass scaler for neural network)
    models, metrics, results_df = train_all_models(X_train, X_test, y_train, y_test, scaler)
    
    print("\n" + "="*70)
    print("✓ Training completed successfully!")
    print("="*70)
    return models, metrics, results_df


if __name__ == "__main__":
    main()
