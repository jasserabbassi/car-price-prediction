"""
Prediction History & Data Quality Module
Track predictions, export results, and generate quality reports
"""

import json
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class PredictionHistory:
    """Track and manage prediction history"""
    
    def __init__(self, history_file='prediction_history.json'):
        self.history_file = Path(history_file)
        self.predictions = self._load_history()
    
    def _load_history(self):
        """Load prediction history from file"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def add_prediction(self, features: Dict, predictions: Dict, model_names: List[str]):
        """
        Add prediction to history
        
        Args:
            features: Input features
            predictions: Predictions from models {model_name: price}
            model_names: List of model names
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'predictions': predictions,
            'models': model_names,
            'average_prediction': sum(predictions.values()) / len(predictions)
        }
        self.predictions.append(record)
        self._save_history()
    
    def _save_history(self):
        """Save history to JSON file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)
    
    def export_csv(self, output_file='prediction_results.csv'):
        """Export prediction history as CSV"""
        if not self.predictions:
            return None
        
        rows = []
        for pred in self.predictions:
            row = {
                'Timestamp': pred['timestamp'],
                **pred['features'],
                **{f"{k} (Price)": v for k, v in pred['predictions'].items()},
                'Average Price': pred['average_prediction']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        return df
    
    def export_json(self, output_file='prediction_results.json'):
        """Export as JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)
        return self.predictions
    
    def get_statistics(self):
        """Get prediction statistics"""
        if not self.predictions:
            return None
        
        all_prices = [p['average_prediction'] for p in self.predictions]
        return {
            'total_predictions': len(all_prices),
            'average_price': sum(all_prices) / len(all_prices),
            'min_price': min(all_prices),
            'max_price': max(all_prices),
            'std_dev': (sum((x - sum(all_prices)/len(all_prices))**2 for x in all_prices) / len(all_prices))**0.5
        }


class DataQualityReport:
    """Generate comprehensive data quality report"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {}
    
    def generate_report(self):
        """Generate complete quality report"""
        self.report = {
            'dataset_overview': self._dataset_overview(),
            'missing_values': self._missing_values(),
            'duplicates': self._duplicates(),
            'outliers': self._outliers(),
            'statistical_summary': self._statistical_summary(),
            'data_types': self._data_types(),
            'quality_score': self._quality_score()
        }
        return self.report
    
    def _dataset_overview(self):
        """Basic dataset info"""
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def _missing_values(self):
        """Analyze missing values"""
        missing = self.df.isnull().sum()
        return {
            'missing_count': missing[missing > 0].to_dict(),
            'missing_percentage': (missing / len(self.df) * 100)[missing > 0].to_dict(),
            'total_missing_cells': missing.sum()
        }
    
    def _duplicates(self):
        """Analyze duplicates"""
        return {
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_percentage': (self.df.duplicated().sum() / len(self.df) * 100)
        }
    
    def _outliers(self):
        """Detect outliers using IQR method"""
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': float(outlier_count / len(self.df) * 100),
                    'bounds': [float(lower_bound), float(upper_bound)]
                }
        
        return outliers
    
    def _statistical_summary(self):
        """Statistical summary of numeric columns"""
        return self.df.describe().to_dict()
    
    def _data_types(self):
        """Data type distribution"""
        return self.df.dtypes.astype(str).value_counts().to_dict()
    
    def _quality_score(self):
        """Calculate overall data quality score (0-100)"""
        score = 100
        
        # Penalize missing values
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        score -= missing_pct * 2
        
        # Penalize duplicates
        dup_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        score -= dup_pct
        
        return max(0, min(100, score))
    
    def to_dict(self):
        """Return report as dictionary"""
        if not self.report:
            self.generate_report()
        return self.report
    
    def to_json(self, output_file='data_quality_report.json'):
        """Export report as JSON"""
        if not self.report:
            self.generate_report()
        
        # Convert to JSON-serializable format
        report_copy = json.loads(json.dumps(self.report, default=str))
        
        with open(output_file, 'w') as f:
            json.dump(report_copy, f, indent=2)
        
        return report_copy


class ConfidenceInterval:
    """Calculate prediction confidence intervals"""
    
    @staticmethod
    def calculate_ci(predictions: List[float], confidence=0.95):
        """
        Calculate confidence interval from multiple model predictions
        
        Args:
            predictions: List of predictions from different models
            confidence: Confidence level (default 0.95 = 95%)
            
        Returns:
            dict: Mean, std, and CI bounds
        """
        mean = sum(predictions) / len(predictions)
        
        # Calculate standard deviation
        variance = sum((x - mean)**2 for x in predictions) / len(predictions)
        std = variance ** 0.5
        
        # Calculate margin of error (using t-distribution approximation)
        n = len(predictions)
        margin_error = 1.96 * std / (n ** 0.5)  # 95% CI
        
        return {
            'mean': mean,
            'std': std,
            'lower_bound': mean - margin_error,
            'upper_bound': mean + margin_error,
            'margin_of_error': margin_error,
            'confidence_level': f"{confidence*100}%"
        }
