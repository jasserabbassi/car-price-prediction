"""
Model Explainability Module - SHAP Values & Feature Attribution
Uses SHAP (SHapley Additive exPlanations) for model interpretability
"""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path


class ModelExplainer:
    """Provides model explainability using SHAP values"""
    
    def __init__(self, model, X_train, feature_names):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            X_train: Training data for SHAP background
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self):
        """Create SHAP explainer based on model type"""
        try:
            # Use TreeExplainer for tree-based models
            if hasattr(self.model, 'predict'):
                self.explainer = shap.TreeExplainer(self.model)
            return True
        except Exception as e:
            print(f"Error creating explainer: {e}")
            return False
    
    def explain_prediction(self, X_instance):
        """
        Get SHAP explanation for a single prediction
        
        Args:
            X_instance: Single instance to explain (shape: 1, n_features)
            
        Returns:
            dict: SHAP values and base values
        """
        if self.explainer is None:
            self.create_explainer()
        
        shap_values = self.explainer.shap_values(X_instance)
        base_value = self.explainer.expected_value
        
        return {
            'shap_values': shap_values,
            'base_value': base_value,
            'features': self.feature_names
        }
    
    def get_feature_importance(self, X_data=None):
        """
        Get mean absolute SHAP values (feature importance)
        
        Args:
            X_data: Data to calculate importance (uses training data if None)
            
        Returns:
            pd.DataFrame: Feature importance ranked
        """
        if self.explainer is None:
            self.create_explainer()
        
        X_eval = X_data if X_data is not None else self.X_train
        shap_values = self.explainer.shap_values(X_eval)
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def create_force_plot(self, X_instance, model_output):
        """Create force plot explanation"""
        if self.explainer is None:
            self.create_explainer()
        
        shap_values = self.explainer.shap_values(X_instance)
        return {
            'shap_values': shap_values[0] if isinstance(shap_values, list) else shap_values,
            'base_value': self.explainer.expected_value,
            'X': X_instance,
            'feature_names': self.feature_names
        }


def load_explainer_data(model_path, scaler_path, encoder_path, X_train):
    """Load model and create explainer"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    
    # Get feature names
    feature_names = get_feature_names()
    
    explainer = ModelExplainer(model, X_train, feature_names)
    return explainer, model, scaler, encoder


def get_feature_names():
    """Get feature names in correct order"""
    return [
        'Brand', 'Model', 'Year', 'Engine Size', 'Mileage',
        'Fuel Type', 'Transmission', 'Condition'
    ]
