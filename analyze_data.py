"""Comprehensive analysis of training data to diagnose overestimation issues."""

import numpy as np
import pandas as pd
from scipy import stats

DATA_FILE = "data/car_price_prediction_.csv"

df = pd.read_csv(DATA_FILE)
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Price distribution
print("\n=== PRICE DISTRIBUTION ===")
print(df['Price'].describe())
print(f"\nPrice skewness: {stats.skew(df['Price']):.3f}")
print(f"Price kurtosis: {stats.kurtosis(df['Price']):.3f}")

# Cheap car representation
for threshold in [5000, 8000, 10000, 15000, 20000]:
    count = (df['Price'] < threshold).sum()
    pct = count / len(df) * 100
    print(f"Cars under ${threshold:,}: {count:,} ({pct:.1f}%)")

# Price percentiles
print("\nPrice percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: ${df['Price'].quantile(p/100):,.0f}")

# Outliers above $50k
high_end = df[df['Price'] > 50000]
print(f"\nCars above $50,000: {len(high_end):,} ({len(high_end)/len(df)*100:.1f}%)")
print(f"  Max price: ${df['Price'].max():,.0f}")

# Mileage distribution
print("\n=== MILEAGE DISTRIBUTION ===")
print(df['Mileage'].describe())
print(f"Mileage skewness: {stats.skew(df['Mileage']):.3f}")

# Correlation with price
print("\n=== CORRELATIONS WITH PRICE ===")
numeric_cols = ['Year', 'Engine Size', 'Mileage', 'Price']
for col in numeric_cols:
    if col != 'Price':
        corr = df[col].corr(df['Price'])
        print(f"  {col}: {corr:.4f}")

# Price by brand
print("\n=== PRICE BY BRAND (top 10) ===")
brand_stats = df.groupby('Brand').agg(
    count=('Price', 'count'),
    mean=('Price', 'mean'),
    median=('Price', 'median'),
    min=('Price', 'min'),
    max=('Price', 'max')
).sort_values('mean', ascending=False)
print(brand_stats.head(10).to_string())

# Ford Fiesta specific analysis
if 'Model' in df.columns:
    fiesta = df[df['Model'].str.contains('Fiesta', case=False, na=False)]
    print(f"\n=== FORD FIESTA SAMPLES ===")
    print(f"Count: {len(fiesta)}")
    if len(fiesta) > 0:
        print(fiesta[['Brand', 'Model', 'Year', 'Mileage', 'Engine Size', 'Price']].head(20).to_string())
        print(f"\nFiesta price stats:")
        print(fiesta['Price'].describe())

# Old high-mileage budget cars
old_cars = df[(df['Year'] < 2017) & (df['Mileage'] > 60000) & (df['Price'] < 15000)]
print(f"\n=== OLD + HIGH MILEAGE + BUDGET CARS ===")
print(f"Count: {len(old_cars):,} ({len(old_cars)/len(df)*100:.1f}%)")
if len(old_cars) > 0:
    print(old_cars[['Brand', 'Model', 'Year', 'Mileage', 'Engine Size', 'Price']].head(20).to_string())

# Log-transform analysis
print("\n=== LOG TRANSFORM ANALYSIS ===")
log_price = np.log1p(df['Price'])
print(f"Original price skewness: {stats.skew(df['Price']):.3f}")
print(f"Log(price) skewness: {stats.skew(log_price):.3f}")
print(f"Log(price) looks more normal: {abs(stats.skew(log_price)) < abs(stats.skew(df['Price']))}")
