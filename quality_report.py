"""
Data Quality Report Module - Comprehensive Analysis
"""

import pandas as pd
import json
from pathlib import Path


def generate_quality_report(data='data/car_price_prediction_.csv'):
    """Generate comprehensive data quality report"""
    
    # Accept both DataFrame and file path
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.read_csv(data)
    
    report = {
        'Dataset Overview': {
            'Total Records': len(df),
            'Total Features': len(df.columns),
            'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        },
        'Feature Analysis': {},
        'Data Quality Metrics': {},
        'Statistical Summary': {}
    }
    
    # Analyze each feature
    for col in df.columns:
        feature_info = {
            'Data Type': str(df[col].dtype),
            'Non-Null Count': int(df[col].notna().sum()),
            'Null Count': int(df[col].isna().sum()),
            'Unique Values': int(df[col].nunique())
        }
        
        if df[col].dtype in ['float64', 'int64']:
            feature_info['Statistics'] = {
                'Mean': round(float(df[col].mean()), 2),
                'Std Dev': round(float(df[col].std()), 2),
                'Min': round(float(df[col].min()), 2),
                'Max': round(float(df[col].max()), 2),
                'Q1': round(float(df[col].quantile(0.25)), 2),
                'Median': round(float(df[col].median()), 2),
                'Q3': round(float(df[col].quantile(0.75)), 2)
            }
        
        report['Feature Analysis'][col] = feature_info
    
    # Data quality metrics
    report['Data Quality Metrics'] = {
        'Duplicate Rows': int(df.duplicated().sum()),
        'Null Values': int(df.isna().sum().sum()),
        'Completeness %': round((1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
    }
    
    # Overall quality score
    completeness = (1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    duplicates_pct = (df.duplicated().sum() / len(df)) * 100
    quality_score = completeness - duplicates_pct
    
    report['Quality Score'] = round(max(0, min(100, quality_score)), 2)
    
    return report
