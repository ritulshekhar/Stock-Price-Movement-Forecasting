import yaml
import os

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_fused_features(price_path, sentiment_path):
    """Combine price and sentiment features for final model"""
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    sentiment = pd.read_csv(sentiment_path, parse_dates=['date'])
    
    # Merge and forward-fill sentiment
    merged = prices.join(sentiment.set_index('date'))
    merged['sentiment'] = merged['sentiment'].fillna(method='ffill').fillna(0.5)
    
    # Create lagged features
    for i in range(1, 4):
        merged[f'price_lag_{i}'] = merged['price'].shift(i)
        merged[f'sentiment_lag_{i}'] = merged['sentiment'].shift(i)
    
    return merged.dropna()