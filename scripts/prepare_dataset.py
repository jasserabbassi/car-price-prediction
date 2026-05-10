"""Download and clean the real Kaggle 'Car Price Prediction Challenge' dataset.

Source: deepcontractor / Kaggle, mirrored on GitHub.
Output: data/car_price_kaggle_clean.csv (~15,000 rows, miles instead of km).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_PATH = DATA_DIR / "car_price_kaggle_raw.csv"
CLEAN_PATH = DATA_DIR / "car_price_kaggle_clean.csv"

# Public mirror of the Kaggle "Car Price Prediction Challenge" dataset.
SOURCE_URL = (
    "https://raw.githubusercontent.com/yukito0209/"
    "is6400-business-data-analytics/main/data/car_price_prediction.csv"
)

# Cleaning bounds (match thesis Table 10).
YEAR_MIN, YEAR_MAX = 1990, 2023
PRICE_MIN, PRICE_MAX = 1_000, 105_000
ENGINE_MIN, ENGINE_MAX = 0.6, 8.0
MILEAGE_MIN_MI, MILEAGE_MAX_MI = 150, 298_000
KM_TO_MI = 0.621371
Z_THRESHOLD = 3.0


def download_raw() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        print(f"[prepare_dataset] reusing cached raw CSV at {RAW_PATH}")
        return RAW_PATH
    print(f"[prepare_dataset] downloading raw CSV from {SOURCE_URL} …")
    resp = requests.get(SOURCE_URL, timeout=60)
    resp.raise_for_status()
    RAW_PATH.write_bytes(resp.content)
    print(f"[prepare_dataset] saved {len(resp.content) / 1024:.0f} KB to {RAW_PATH}")
    return RAW_PATH


def parse_mileage(value) -> float:
    """Kaggle stores mileage as e.g. '186005 km'. Return numeric km."""
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return np.nan
    text = str(value).lower().replace(",", "").strip()
    if text in {"", "nan", "none"}:
        return np.nan
    text = text.replace("km", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_engine(value) -> float:
    """Kaggle 'Engine volume' is e.g. '2.0' or '2.0 Turbo'. Strip suffixes."""
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return np.nan
    text = str(value).lower().replace("turbo", "").strip()
    if text in {"", "nan", "none"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[prepare_dataset] raw rows: {len(df):,}")

    # Normalise column names to match the thesis / project conventions.
    df = df.rename(
        columns={
            "Engine volume": "Engine Size",
            "Prod. year": "Year",
            "Manufacturer": "Brand",
            "Category": "Condition",  # use Category as our 'Condition'-like field
            "Fuel type": "Fuel Type",
            "Gear box type": "Transmission",
        }
    )

    # Parse numeric fields.
    df["Engine Size"] = df["Engine Size"].apply(parse_engine)
    df["Mileage_km"] = df["Mileage"].apply(parse_mileage)
    df["Mileage"] = (df["Mileage_km"] * KM_TO_MI).round().astype("Int64")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Apply bounds.
    df = df[
        df["Year"].between(YEAR_MIN, YEAR_MAX)
        & df["Price"].between(PRICE_MIN, PRICE_MAX)
        & df["Engine Size"].between(ENGINE_MIN, ENGINE_MAX)
        & df["Mileage"].between(MILEAGE_MIN_MI, MILEAGE_MAX_MI)
    ].copy()
    print(f"[prepare_dataset] after range filter: {len(df):,}")

    # Drop rows missing any required column.
    required = ["Brand", "Model", "Year", "Engine Size", "Fuel Type",
                "Transmission", "Mileage", "Condition", "Price"]
    df = df.dropna(subset=required)
    print(f"[prepare_dataset] after dropna: {len(df):,}")

    # Z-score outlier removal on numeric columns.
    numeric_cols = ["Year", "Engine Size", "Mileage", "Price"]
    for col in numeric_cols:
        col_data = df[col].astype(float)
        z = np.abs((col_data - col_data.mean()) / col_data.std(ddof=0))
        df = df[z < Z_THRESHOLD]
    print(f"[prepare_dataset] after Z-score filter: {len(df):,}")

    # Final column ordering for downstream pipeline.
    df = df.reset_index(drop=True)
    df.insert(0, "Car ID", np.arange(1, len(df) + 1))
    df = df[
        [
            "Car ID", "Brand", "Year", "Engine Size", "Fuel Type",
            "Transmission", "Mileage", "Condition", "Price", "Model",
        ]
    ]
    return df


def main() -> int:
    raw_path = download_raw()
    raw = pd.read_csv(raw_path)
    cleaned = clean(raw)

    print("\n[prepare_dataset] summary statistics:")
    print(cleaned.describe(include="all").T[["count", "mean", "std", "min", "max"]])

    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"\n[prepare_dataset] wrote {len(cleaned):,} rows to {CLEAN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
