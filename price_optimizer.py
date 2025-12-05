import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt


def load_data(history_path):
    df = pd.read_csv(history_path, parse_dates=["date"]) 
    df = df.sort_values("date").reset_index(drop=True)
    return df


def engineer_features(df):
    df = df.copy()
    # competitor mean and best
    df['comp_mean'] = df[['comp1_price','comp2_price','comp3_price']].mean(axis=1)
    df['comp_min'] = df[['comp1_price','comp2_price','comp3_price']].min(axis=1)

    # price differentials
    df['price_minus_comp_mean'] = df['price'] - df['comp_mean']
    df['price_minus_comp_min'] = df['price'] - df['comp_min']

    # lag features of price and volume
    for lag in [1,2,3,7]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

    # rolling features
    df['volume_roll_7'] = df['volume'].rolling(7, min_periods=1).mean().shift(1)
    df['price_roll_7'] = df['price'].rolling(7, min_periods=1).mean().shift(1)

    # day of week (seasonality)
    df['dow'] = df['date'].dt.dayofweek

    # drop rows with NaNs introduced by shifting
    df = df.dropna().reset_index(drop=True)
    return df


def prepare_training_data(df):
    df_feat = engineer_features(df)
    feature_cols = [c for c in df_feat.columns if c not in ['date','volume']]
    X = df_feat[feature_cols]
    y = df_feat['volume']
    return X, y, feature_cols, df_feat


def train_model(X, y):
    # simple time-series split to evaluate
    tscv = TimeSeriesSplit(n_splits=3)
    rmses = []
    maes = []
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        rmses.append(np.sqrt(mean_squared_error(yte, preds)))
        maes.append(mean_absolute_error(yte, preds))
    # final fit on full data
    model.fit(X, y)
    return model, np.mean(rmses), np.mean(maes)


def build_today_features(today_json, feature_cols, last_rows):
    # last_rows: recent rows from historical processed df to get lags/rolls
    today = today_json.copy()
    # compute competitor mean/min
    comp_prices = [today['comp1_price'], today['comp2_price'], today['comp3_price']]
    comp_mean = np.mean(comp_prices)
    comp_min = np.min(comp_prices)

    base = {}
    base['comp_mean'] = comp_mean
    base['comp_min'] = comp_min

    # we'll fill price and price lags: note that model expects columns for a given candidate price
    # get last price/volume lags from history
    for lag in [1,2,3,7]:
        colp = f'price_lag_{lag}'
        colv = f'volume_lag_{lag}'
        base[colp] = last_rows.iloc[-lag]['price'] if len(last_rows) >= lag else last_rows['price'].iloc[-1]
        base[colv] = last_rows.iloc[-lag]['volume'] if len(last_rows) >= lag else last_rows['volume'].iloc[-1]

    base['volume_roll_7'] = last_rows['volume'].rolling(7, min_periods=1).mean().iloc[-1]
    base['price_roll_7'] = last_rows['price'].rolling(7, min_periods=1).mean().iloc[-1]
    base['dow'] = pd.to_datetime(today['date']).dayofweek

    # We'll create a function that, given a candidate price, returns a feature vector
    def features_for_price(candidate_price):
        feat = base.copy()
        feat['price'] = candidate_price
        feat['price_minus_comp_mean'] = candidate_price - feat['comp_mean']
        feat['price_minus_comp_min'] = candidate_price - feat['comp_min']
        # ensure order of columns matches feature_cols
        row = [feat.get(c, np.nan) for c in feature_cols]
        return np.array(row, dtype=float).reshape(1, -1)

    return features_for_price


def optimize_price(model, features_for_price, today_json, feature_cols, guardrails):
    last_price = today_json['price']
    cost = today_json['cost']

    # build candidate range respecting guardrails
    max_change = guardrails.get('max_abs_change', 2.0)
    min_margin = guardrails.get('min_margin', 0.5)  # rupees
    # candidate bounds
    low = max(cost + min_margin, last_price - max_change)
    high = last_price + max_change
    if low >= high:
        # fallback: allow small range above cost
        low = cost + min_margin
        high = low + 2.0

    candidates = np.linspace(low, high, 101)

    best = None
    best_profit = -np.inf
    best_volume = None

    for p in candidates:
        x = features_for_price(p)
        pred_vol = model.predict(x)[0]
        profit = (p - cost) * pred_vol
        # apply competitor alignment penalty (optional): discourage pricing far above market
        # (here we keep it simple)
        if profit > best_profit:
            best_profit = profit
            best = p
            best_volume = pred_vol

    return {
        'recommended_price': float(round(best, 2)),
        'expected_volume': float(round(best_volume, 2)),
        'expected_profit': float(round(best_profit, 2)),
        'candidate_low': float(round(low,2)),
        'candidate_high': float(round(high,2))
    }


def main(args):
    history_path = Path(args.history)
    today_path = Path(args.today)

    # load files
    df = load_data(history_path)
    with open(today_path, 'r') as f:
        today_json = json.load(f)

    # prepare training
    X, y, feature_cols, df_feat = prepare_training_data(df)
    print(f"Training samples: {len(X)}. Features: {len(feature_cols)}")

    model, rmse, mae = train_model(X, y)
    print(f"Trained model. CV RMSE: {rmse:.2f}, CV MAE: {mae:.2f}")

    # save model
    joblib.dump({'model': model, 'feature_cols': feature_cols}, 'price_model.joblib')
    print("Saved model to price_model.joblib")

    # build today's feature function
    features_for_price = build_today_features(today_json, feature_cols, df)

    # guardrails: you can tune these
    guardrails = {
        'max_abs_change': args.max_change,
        'min_margin': args.min_margin
    }

    result = optimize_price(model, features_for_price, today_json, feature_cols, guardrails)

    print('\n=== Recommendation ===')
    print(f"Today's input (from file): {today_json}")
    print(f"Search range: {result['candidate_low']} to {result['candidate_high']}")
    print(f"Recommended price: {result['recommended_price']}")
    print(f"Expected volume (L): {result['expected_volume']}")
    print(f"Expected profit (approx): {result['expected_profit']}")

    # Optional: quick diagnostic plot of true vs predicted on last split
    try:
        # predict on last 90 days
        recent = df_feat.tail(90)
        Xr = recent[feature_cols]
        preds = model.predict(Xr)
        plt.figure(figsize=(8,4))
        plt.plot(recent['date'], recent['volume'], label='actual')
        plt.plot(recent['date'], preds, label='predicted')
        plt.xlabel('date')
        plt.ylabel('volume')
        plt.title('Actual vs Predicted (last 90 days)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('diagnostic_actual_vs_predicted.png')
        print('Saved diagnostic plot to diagnostic_actual_vs_predicted.png')
    except Exception as e:
        print('Could not create diagnostic plot:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, required=True, help='path to oil_retail_history.csv')
    parser.add_argument('--today', type=str, required=True, help='path to today_example.json')
    parser.add_argument('--max_change', type=float, default=2.0, help='max absolute price change allowed (Rs)')
    parser.add_argument('--min_margin', type=float, default=0.5, help='minimum profit margin (price - cost) in Rs')
    args = parser.parse_args()
    main(args)
