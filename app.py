import gradio as gr
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

# Load model and data on startup
try:
    model_data = joblib.load('price_model.joblib')
    historical_df = pd.read_csv('oil_retail_history.csv', parse_dates=['date'])
    print("Model and data loaded successfully")
except Exception as e:
    print(f"Error loading model/data: {e}")
    model_data = None
    historical_df = None

def build_features_for_price(date, current_price, cost, comp1, comp2, comp3, candidate_price):
    comp_mean = np.mean([comp1, comp2, comp3])
    comp_min = np.min([comp1, comp2, comp3])
    
    # Get recent data for lags
    recent = historical_df.tail(10)
    
    # Build feature vector
    features = {
        'price': candidate_price,
        'comp_mean': comp_mean,
        'comp_min': comp_min,
        'price_minus_comp_mean': candidate_price - comp_mean,
        'price_minus_comp_min': candidate_price - comp_min,
        'dow': pd.to_datetime(date).dayofweek
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

def optimize_price(date, current_price, cost, comp1_price, comp2_price, comp3_price, max_change, min_margin):
    if model_data is None:
        return "Error: Model not loaded", "", "", "", ""
    
    try:
        model = model_data['model']
        
        # Calculate price bounds
        low = max(cost + min_margin, current_price - max_change)
        high = current_price + max_change
        
        if low >= high:
            low = cost + min_margin
            high = low + 2.0
        
        # Find optimal price
        candidates = np.linspace(low, high, 51)
        best_price = None
        best_profit = -np.inf
        best_volume = None
        
        for price in candidates:
            features = build_features_for_price(date, current_price, cost, comp1_price, comp2_price, comp3_price, price)
            volume = model.predict(features)[0]
            profit = (price - cost) * volume
            
            if profit > best_profit:
                best_profit = profit
                best_price = price
                best_volume = volume
        
        return (
            f"₹{best_price:.2f}",
            f"{best_volume:.0f} L",
            f"₹{best_profit:.0f}",
            f"₹{low:.2f} - ₹{high:.2f}",
            "✅ Optimization completed successfully"
        )
        
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", ""

# Create Gradio interface
with gr.Blocks(title="Oil Price Optimizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛢️ Oil Price Optimizer")
    gr.Markdown("ML-powered price optimization for oil retail using demand prediction and profit maximization")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Parameters")
            date = gr.Textbox(label="Date", value="2024-12-31", placeholder="YYYY-MM-DD")
            current_price = gr.Number(label="Current Price (₹)", value=94.45)
            cost = gr.Number(label="Cost (₹)", value=85.77)
            
            gr.Markdown("### Competitor Prices")
            comp1_price = gr.Number(label="Competitor 1 Price (₹)", value=95.01)
            comp2_price = gr.Number(label="Competitor 2 Price (₹)", value=95.70)
            comp3_price = gr.Number(label="Competitor 3 Price (₹)", value=95.21)
            
            gr.Markdown("### Business Constraints")
            max_change = gr.Number(label="Max Price Change (₹)", value=2.0)
            min_margin = gr.Number(label="Min Margin (₹)", value=0.5)
            
            optimize_btn = gr.Button("🎯 Optimize Price", variant="primary")
        
        with gr.Column():
            gr.Markdown("### Optimization Results")
            recommended_price = gr.Textbox(label="Recommended Price", interactive=False)
            expected_volume = gr.Textbox(label="Expected Volume", interactive=False)
            expected_profit = gr.Textbox(label="Expected Profit", interactive=False)
            price_range = gr.Textbox(label="Search Range", interactive=False)
            status = gr.Textbox(label="Status", interactive=False)
    
    optimize_btn.click(
        optimize_price,
        inputs=[date, current_price, cost, comp1_price, comp2_price, comp3_price, max_change, min_margin],
        outputs=[recommended_price, expected_volume, expected_profit, price_range, status]
    )
    
    gr.Markdown("### How it works")
    gr.Markdown("""
    1. **Input today's market data** - current price, cost, and competitor prices
    2. **Set business constraints** - maximum price change and minimum margin
    3. **ML model predicts demand** - using RandomForest with engineered features
    4. **Optimization finds best price** - maximizes profit within constraints
    """)

if __name__ == "__main__":
    demo.launch()