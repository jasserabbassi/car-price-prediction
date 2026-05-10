"""
FastAPI Model Server - Production API for Car Price Prediction
Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Load models and preprocessing objects
with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('models/gradient_boosting_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

app = FastAPI(
    title="Car Price Prediction API",
    description="Advanced ML API for car price prediction with multiple models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CarData(BaseModel):
    """Car data input model"""
    Brand: str
    Model: str
    Year: int
    Engine_Size: float
    Mileage: int
    Fuel_Type: str
    Transmission: str
    Condition: str


class PredictionResponse(BaseModel):
    """API response model"""
    random_forest: float
    xgboost: float
    gradient_boosting: float
    ensemble_average: float
    confidence_range: dict
    timestamp: str


@app.get("/")
async def root():
    """API welcome message"""
    return {
        "message": "Car Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model-info",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": ["RandomForest", "XGBoost", "GradientBoosting"]
    }


@app.get("/model-info")
async def model_info():
    """Get information about loaded models"""
    return {
        "models": {
            "random_forest": {
                "type": "RandomForestRegressor",
                "n_estimators": 200,
                "max_depth": 20
            },
            "xgboost": {
                "type": "XGBRegressor",
                "n_estimators": 150,
                "max_depth": 7
            },
            "gradient_boosting": {
                "type": "GradientBoostingRegressor",
                "n_estimators": 200,
                "learning_rate": 0.1
            }
        },
        "features": [
            "Brand", "Model", "Year", "Engine_Size", 
            "Mileage", "Fuel_Type", "Transmission", "Condition"
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(car_data: CarData):
    """
    Predict car price from single input
    
    Example:
    {
        "Brand": "Toyota",
        "Model": "Corolla",
        "Year": 2020,
        "Engine_Size": 1.8,
        "Mileage": 45000,
        "Fuel_Type": "Petrol",
        "Transmission": "Automatic",
        "Condition": "Good"
    }
    """
    try:
        # Prepare data
        X = prepare_input(car_data.dict())
        
        # Make predictions
        rf_pred = float(rf_model.predict(X)[0])
        xgb_pred = float(xgb_model.predict(X)[0])
        gb_pred = float(gb_model.predict(X)[0])
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + xgb_pred + gb_pred) / 3
        
        # Confidence interval
        predictions = [rf_pred, xgb_pred, gb_pred]
        mean = sum(predictions) / len(predictions)
        std = (sum((x - mean)**2 for x in predictions) / len(predictions)) ** 0.5
        margin = 1.96 * std / (len(predictions) ** 0.5)
        
        return PredictionResponse(
            random_forest=rf_pred,
            xgboost=xgb_pred,
            gradient_boosting=gb_pred,
            ensemble_average=ensemble_pred,
            confidence_range={
                "lower": ensemble_pred - margin,
                "upper": ensemble_pred + margin,
                "margin": margin
            },
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(cars: List[CarData]):
    """
    Predict prices for multiple cars at once
    
    Args:
        cars: List of car data objects
    
    Returns:
        List of predictions with ensemble averages
    """
    try:
        results = []
        
        for car_data in cars:
            X = prepare_input(car_data.dict())
            
            rf_pred = float(rf_model.predict(X)[0])
            xgb_pred = float(xgb_model.predict(X)[0])
            gb_pred = float(gb_model.predict(X)[0])
            ensemble_pred = (rf_pred + xgb_pred + gb_pred) / 3
            
            results.append({
                "input": car_data.dict(),
                "predictions": {
                    "random_forest": rf_pred,
                    "xgboost": xgb_pred,
                    "gradient_boosting": gb_pred,
                    "ensemble_average": ensemble_pred
                }
            })
        
        return {
            "count": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def prepare_input(car_dict: dict) -> np.ndarray:
    """
    Prepare input data for prediction
    
    Args:
        car_dict: Input car data dictionary
        
    Returns:
        Processed numpy array ready for model prediction
    """
    # Create DataFrame with single row
    df = pd.DataFrame([car_dict])
    
    # Rename columns to match training format
    df.columns = ['Brand', 'Model', 'Year', 'Engine Size', 'Mileage', 
                  'Fuel Type', 'Transmission', 'Condition']
    
    # Encode categorical features
    categorical_features = ['Brand', 'Model', 'Fuel Type', 'Transmission', 'Condition']
    df[categorical_features] = encoder.transform(df[categorical_features])
    
    # Scale numerical features
    numerical_features = ['Year', 'Engine Size', 'Mileage']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    return df.values


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
