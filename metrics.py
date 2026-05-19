"""Model evaluation and comparison metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Adjusted R²
    n = len(y_true)
    p = 1  # number of features (simplified)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Adj_R2': adj_r2,
        'MAPE': mape,
        'Mean_Actual': np.mean(y_true),
        'Std_Actual': np.std(y_true)
    }


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Plot actual vs predicted values."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(size=8, color='#1f77b4', opacity=0.6),
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='red', width=2),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=f"{model_name}: Actual vs Predicted Prices",
        xaxis_title="Actual Price ($)",
        yaxis_title="Predicted Price ($)",
        template="plotly_white",
        height=400,
        hovermode="closest"
    )
    
    return fig


def plot_residuals(y_true, y_pred, model_name):
    """Plot residuals."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(size=8, color='#ff7f0e', opacity=0.6),
        name='Residuals'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f"{model_name}: Residual Plot",
        xaxis_title="Predicted Price ($)",
        yaxis_title="Residuals ($)",
        template="plotly_white",
        height=400,
        hovermode="closest"
    )
    
    return fig


def plot_residual_distribution(y_true, y_pred, model_name):
    """Plot residual distribution."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        name='Residuals',
        marker_color='#2ca02c',
        opacity=0.7
    ))
    
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Zero Error"
    )
    
    fig.update_layout(
        title=f"{model_name}: Residual Distribution",
        xaxis_title="Residuals ($)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_metrics_comparison_table(models_metrics):
    """Create comparison table of model metrics."""
    df = pd.DataFrame(models_metrics).T
    
    # Format for display
    display_cols = ['R2', 'Adj_R2', 'RMSE', 'MAE', 'MAPE']
    for col in display_cols:
        if col in df.columns:
            if col in ['R2', 'Adj_R2', 'MAPE']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if x < 1 else f"{x:.2%}")
            else:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}")
    
    return df[display_cols]


def plot_metrics_comparison(models_metrics):
    """Plot metrics comparison across models."""
    df = pd.DataFrame(models_metrics).T
    
    fig = go.Figure()
    
    # R² Score
    fig.add_trace(go.Bar(
        name='R² Score',
        x=df.index,
        y=df['R2'],
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title="Model R² Score Comparison",
        xaxis_title="Model",
        yaxis_title="R² Score",
        template="plotly_white",
        height=400,
        hovermode="x"
    )
    
    return fig


def plot_rmse_comparison(models_metrics):
    """Plot RMSE comparison."""
    df = pd.DataFrame(models_metrics).T
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='RMSE',
        x=df.index,
        y=df['RMSE'],
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title="Model RMSE Comparison (Lower is Better)",
        xaxis_title="Model",
        yaxis_title="RMSE ($)",
        template="plotly_white",
        height=400,
        hovermode="x"
    )
    
    return fig


def create_error_analysis(y_true, y_pred):
    """Create detailed error analysis."""
    errors = np.abs(y_true - y_pred)
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    
    analysis = {
        'Mean Absolute Error': errors.mean(),
        'Median Absolute Error': np.median(errors),
        'Std Dev of Errors': errors.std(),
        'Max Error': errors.max(),
        'Min Error': errors.min(),
        'Mean MAPE': percentage_errors.mean(),
        '90th Percentile Error': np.percentile(errors, 90),
        '95th Percentile Error': np.percentile(errors, 95)
    }
    
    return analysis


def get_prediction_confidence(model, X, confidence_level=0.95):
    """Get confidence intervals for predictions (for ensemble models)."""
    predictions = model.predict(X)
    
    # Simplified confidence calculation using prediction std
    if hasattr(model, 'estimators_'):
        # For ensemble models
        pred_std = np.std([tree.predict(X) for tree in model.estimators_], axis=0)
    else:
        pred_std = predictions * 0.1  # Assume 10% std for single models
    
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    
    lower_bound = predictions - z_score * pred_std
    upper_bound = predictions + z_score * pred_std
    
    return predictions, lower_bound, upper_bound
