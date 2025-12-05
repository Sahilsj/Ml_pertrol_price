# Oil Price Optimizer

ML-based price optimization system for oil retail using demand prediction and profit maximization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run price optimization
python price_optimizer.py --history oil_retail_history.csv --today today_example.json

# Start API server
uvicorn api:app --reload
```

## Features

- **Demand Prediction**: RandomForest model with engineered features (competitor prices, lags, rolling averages)
- **Price Optimization**: Finds optimal price within business guardrails to maximize profit
- **Business Constraints**: Enforces max daily price change and minimum margin requirements
- **API Interface**: FastAPI wrapper for production deployment

## Usage

### Command Line
```bash
python price_optimizer.py --history data.csv --today input.json --max_change 2.0 --min_margin 0.5
```

### API
```bash
# Interactive docs at http://127.0.0.1:8000/docs
# Health check
GET http://127.0.0.1:8000/health

# Price optimization
POST http://127.0.0.1:8000/optimize
{
  "date": "2024-12-31",
  "price": 94.45,
  "cost": 85.77,
  "comp1_price": 95.01,
  "comp2_price": 95.7,
  "comp3_price": 95.21
}
```

## Input Format

**Historical Data (CSV)**: date, price, volume, cost, comp1_price, comp2_price, comp3_price

**Today's Data (JSON)**:
```json
{
  "date": "2024-12-31",
  "price": 94.45,
  "cost": 85.77,
  "comp1_price": 95.01,
  "comp2_price": 95.7,
  "comp3_price": 95.21
}
```

## Output

- Recommended price
- Expected volume and profit
- Model performance metrics
- Diagnostic visualization