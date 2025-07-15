import pandas as pd
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_fused_features(price_path, sentiment_path):
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    sentiment = pd.read_csv(sentiment_path, parse_dates=['date'])
    
    # Merge on date
    sentiment.set_index('date', inplace=True)
    merged = prices.join(sentiment, how='left')
    merged['sentiment'] = merged['sentiment'].fillna(method='ffill').fillna(0.5)

    # Lag features
    for i in range(1, 4):
        merged[f'Close_lag_{i}'] = merged['Close'].shift(i)
        merged[f'sentiment_lag_{i}'] = merged['sentiment'].shift(i)

    return merged.dropna()
