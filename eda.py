"""Exploratory Data Analysis module for car price prediction."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st


def load_and_analyze_data(data_file):
    """Load data and perform basic analysis."""
    df = pd.read_csv(data_file)
    return df


def get_summary_statistics(df):
    """Get summary statistics."""
    return {
        'total_records': len(df),
        'total_features': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }


def plot_price_distribution(df):
    """Plot car price distribution."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['Price'],
        nbinsx=50,
        name='Price Distribution',
        marker_color='#1f77b4',
        opacity=0.7
    ))
    
    fig.add_vline(
        x=df['Price'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${df['Price'].mean():,.0f}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=df['Price'].median(),
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: ${df['Price'].median():,.0f}",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Car Price Distribution",
        xaxis_title="Price ($)",
        yaxis_title="Frequency",
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_price_by_brand(df):
    """Plot price statistics by brand."""
    brand_stats = df.groupby('Brand')['Price'].agg(['mean', 'median', 'count']).reset_index()
    brand_stats = brand_stats.sort_values('mean', ascending=False).head(15)
    
    fig = px.bar(
        brand_stats,
        x='Brand',
        y='mean',
        color='count',
        title="Average Car Price by Brand (Top 15)",
        labels={'mean': 'Average Price ($)', 'count': 'Count'},
        color_continuous_scale='Viridis',
        hover_data={'median': True, 'count': True}
    )
    
    fig.update_layout(height=400, template="plotly_white")
    fig.update_xaxes(tickangle=-45)
    
    return fig


def plot_price_vs_mileage(df):
    """Plot price vs mileage scatter."""
    fig = px.scatter(
        df,
        x='Mileage',
        y='Price',
        color='Year',
        size='Engine Size',
        hover_name='Brand',
        title="Car Price vs Mileage (colored by Year, sized by Engine)",
        labels={'Mileage': 'Mileage (km)', 'Price': 'Price ($)'},
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, template="plotly_white")
    
    return fig


def plot_price_by_year(df):
    """Plot price evolution by year."""
    year_stats = df.groupby('Year').agg({
        'Price': ['mean', 'median', 'count', 'std']
    }).reset_index()
    year_stats.columns = ['Year', 'Mean_Price', 'Median_Price', 'Count', 'Std_Price']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=year_stats['Year'],
        y=year_stats['Mean_Price'],
        name='Mean Price',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=year_stats['Year'],
        y=year_stats['Median_Price'],
        name='Median Price',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Car Price Evolution by Year",
        xaxis_title="Year",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_correlation_heatmap(df):
    """Plot correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap",
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(height=500, template="plotly_white")
    
    return fig


def plot_price_by_fuel_type(df):
    """Plot price by fuel type."""
    fuel_stats = df.groupby('Fuel Type')['Price'].agg(['mean', 'median', 'count', 'std']).reset_index()
    fuel_stats = fuel_stats.sort_values('mean', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=fuel_stats['Fuel Type'],
        y=fuel_stats['mean'],
        name='Mean Price',
        marker_color='#1f77b4',
        error_y=dict(type='data', array=fuel_stats['std'])
    ))
    
    fig.update_layout(
        title="Average Car Price by Fuel Type",
        xaxis_title="Fuel Type",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_engine_size_distribution(df):
    """Plot engine size distribution."""
    fig = px.box(
        df,
        x='Fuel Type',
        y='Engine Size',
        color='Fuel Type',
        title="Engine Size Distribution by Fuel Type",
        labels={'Engine Size': 'Engine Size (L)'},
        points="outliers"
    )
    
    fig.update_layout(height=400, template="plotly_white")
    
    return fig


def plot_transmission_analysis(df):
    """Plot transmission type analysis."""
    trans_stats = df.groupby('Transmission').agg({
        'Price': ['mean', 'count'],
        'Mileage': 'mean'
    }).reset_index()
    trans_stats.columns = ['Transmission', 'AvgPrice', 'Count', 'AvgMileage']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=trans_stats['Transmission'],
        y=trans_stats['AvgPrice'],
        name='Average Price',
        marker_color='#1f77b4',
        yaxis='y1'
    ))
    
    fig.add_trace(go.Scatter(
        x=trans_stats['Transmission'],
        y=trans_stats['Count'],
        name='Count',
        marker_color='#ff7f0e',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Transmission Type Analysis",
        xaxis_title="Transmission Type",
        yaxis=dict(title="Average Price ($)"),
        yaxis2=dict(title="Count", overlaying='y', side='right'),
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def calculate_outliers(df, column, method='iqr'):
    """Detect outliers using IQR method."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return len(outliers), lower_bound, upper_bound
    
    return 0, None, None


def get_statistical_insights(df):
    """Generate statistical insights."""
    insights = {}
    
    # Price insights
    price_skewness = stats.skew(df['Price'])
    price_kurtosis = stats.kurtosis(df['Price'])
    
    insights['price_skewness'] = price_skewness
    insights['price_kurtosis'] = price_kurtosis
    insights['price_range'] = (df['Price'].min(), df['Price'].max())
    insights['price_iqr'] = (df['Price'].quantile(0.25), df['Price'].quantile(0.75))
    
    # Mileage insights
    insights['mileage_range'] = (df['Mileage'].min(), df['Mileage'].max())
    insights['avg_mileage'] = df['Mileage'].mean()
    
    # Brand insights
    insights['total_brands'] = df['Brand'].nunique()
    insights['most_common_brand'] = df['Brand'].value_counts().index[0]
    
    # Year insights
    insights['year_range'] = (int(df['Year'].min()), int(df['Year'].max()))
    
    return insights
