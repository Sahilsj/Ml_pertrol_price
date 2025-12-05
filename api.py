from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

app = FastAPI(title="Oil Price Optimizer API")

class PriceRequest(BaseModel):
    date: str
    price: float
    cost: float
    comp1_price: float
    comp2_price: float
    comp3_price: float
    max_change: float = 2.0
    min_margin: float = 0.5

class PriceResponse(BaseModel):
    recommended_price: float
    expected_volume: float
    expected_profit: float
    candidate_low: float
    candidate_high: float

# Load model and historical data on startup
model_data = None
historical_df = None

@app.on_event("startup")
async def load_model():
    global model_data, historical_df
    try:
        model_data = joblib.load('price_model.joblib')
        historical_df = pd.read_csv('oil_retail_history.csv', parse_dates=['date'])
        print("Model and data loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model/data: {e}")

def build_features_for_price(request: PriceRequest, candidate_price: float):
    # Compute competitor stats
    comp_prices = [request.comp1_price, request.comp2_price, request.comp3_price]
    comp_mean = np.mean(comp_prices)
    comp_min = np.min(comp_prices)
    
    # Get recent data for lags
    recent = historical_df.tail(10)
    
    # Build feature vector
    features = {
        'price': candidate_price,
        'comp_mean': comp_mean,
        'comp_min': comp_min,
        'price_minus_comp_mean': candidate_price - comp_mean,
        'price_minus_comp_min': candidate_price - comp_min,
        'dow': pd.to_datetime(request.date).dayofweek
    }
    
    # Add lag features
    for lag in [1,2,3,7]:
        features[f'price_lag_{lag}'] = recent['price'].iloc[-lag] if len(recent) >= lag else recent['price'].iloc[-1]
        features[f'volume_lag_{lag}'] = recent['volume'].iloc[-lag] if len(recent) >= lag else recent['volume'].iloc[-1]
    
    # Add rolling features
    features['volume_roll_7'] = recent['volume'].rolling(7, min_periods=1).mean().iloc[-1]
    features['price_roll_7'] = recent['price'].rolling(7, min_periods=1).mean().iloc[-1]
    
    # Order features to match model
    feature_cols = model_data['feature_cols']
    row = [features.get(c, 0) for c in feature_cols]
    return np.array(row).reshape(1, -1)

@app.post("/optimize", response_model=PriceResponse)
async def optimize_price(request: PriceRequest):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = model_data['model']
    
    # Calculate price bounds
    low = max(request.cost + request.min_margin, request.price - request.max_change)
    high = request.price + request.max_change
    
    if low >= high:
        low = request.cost + request.min_margin
        high = low + 2.0
    
    # Find optimal price
    candidates = np.linspace(low, high, 51)
    best_price = None
    best_profit = -np.inf
    best_volume = None
    
    for price in candidates:
        features = build_features_for_price(request, price)
        volume = model.predict(features)[0]
        profit = (price - request.cost) * volume
        
        if profit > best_profit:
            best_profit = profit
            best_price = price
            best_volume = volume
    
    return PriceResponse(
        recommended_price=round(best_price, 2),
        expected_volume=round(best_volume, 2),
        expected_profit=round(best_profit, 2),
        candidate_low=round(low, 2),
        candidate_high=round(high, 2)
    )

@app.get("/")
async def root():
    return {"message": "Oil Price Optimizer API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_data is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)