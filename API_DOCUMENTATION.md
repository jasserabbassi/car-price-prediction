# API Documentation

## Car Price Prediction API

Complete REST API documentation for the car price prediction system.

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
**GET** `/health`

Check API status and loaded models.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-31T10:30:00",
  "models_loaded": ["RandomForest", "XGBoost", "GradientBoosting"]
}
```

---

#### 2. Model Information
**GET** `/model-info`

Get information about available models and features.

**Response:**
```json
{
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
  "features": ["Brand", "Model", "Year", "Engine_Size", "Mileage", "Fuel_Type", "Transmission", "Condition"]
}
```

---

#### 3. Single Prediction
**POST** `/predict`

Predict car price from single input.

**Request Body:**
```json
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
```

**Response:**
```json
{
  "random_forest": 25340.50,
  "xgboost": 25890.20,
  "gradient_boosting": 25450.75,
  "ensemble_average": 25560.48,
  "confidence_range": {
    "lower": 25200.30,
    "upper": 25920.66,
    "margin": 360.18
  },
  "timestamp": "2026-03-31T10:30:00"
}
```

**Status Codes:**
- `200`: Prediction successful
- `400`: Invalid input data
- `500`: Server error

---

#### 4. Batch Prediction
**POST** `/batch-predict`

Predict prices for multiple cars at once.

**Request Body:**
```json
[
  {
    "Brand": "Toyota",
    "Model": "Corolla",
    "Year": 2020,
    "Engine_Size": 1.8,
    "Mileage": 45000,
    "Fuel_Type": "Petrol",
    "Transmission": "Automatic",
    "Condition": "Good"
  },
  {
    "Brand": "Honda",
    "Model": "Civic",
    "Year": 2021,
    "Engine_Size": 1.5,
    "Mileage": 25000,
    "Fuel_Type": "Hybrid",
    "Transmission": "Automatic",
    "Condition": "Excellent"
  }
]
```

**Response:**
```json
{
  "count": 2,
  "predictions": [
    {
      "input": {
        "Brand": "Toyota",
        "Model": "Corolla",
        "Year": 2020,
        "Engine_Size": 1.8,
        "Mileage": 45000,
        "Fuel_Type": "Petrol",
        "Transmission": "Automatic",
        "Condition": "Good"
      },
      "predictions": {
        "random_forest": 25340.50,
        "xgboost": 25890.20,
        "gradient_boosting": 25450.75,
        "ensemble_average": 25560.48
      }
    }
  ],
  "timestamp": "2026-03-31T10:30:00"
}
```

---

### Input Parameters

#### Valid Values

**Brands:** Toyota, Honda, BMW, Mercedes, Audi, Ford, Chevrolet, Nissan, etc.

**Fuel Types:** Petrol, Diesel, Hybrid, Electric

**Transmission:** Manual, Automatic, CVT

**Condition:** Excellent, Good, Fair, Poor

**Year Range:** 2000-2025

**Engine Size:** 0.5 - 6.0 (in liters)

**Mileage:** 0 - 500,000 (in kilometers)

---

### Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

# Single prediction
car_data = {
    "Brand": "Toyota",
    "Model": "Corolla",
    "Year": 2020,
    "Engine_Size": 1.8,
    "Mileage": 45000,
    "Fuel_Type": "Petrol",
    "Transmission": "Automatic",
    "Condition": "Good"
}

response = requests.post(f"{API_URL}/predict", json=car_data)
result = response.json()

print(f"Predicted Price: ${result['ensemble_average']:,.2f}")
print(f"Range: ${result['confidence_range']['lower']:,.2f} - ${result['confidence_range']['upper']:,.2f}")
```

---

### Error Handling

**Bad Request Example:**
```json
{
  "detail": "Invalid fuel type: 'LPG'"
}
```

**Response:**
- Status Code: 400
- Message describes the validation error

---

### Rate Limiting

- No rate limiting in development
- Production: 100 requests per minute per IP

---

### Authentication

- Currently no authentication required
- In production, add API key authentication

---

### CORS Policy

- Allow all origins in development
- Restrict in production

### Deployment

#### Using Gunicorn (Production)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 api:app
```

#### Using Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```
