import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']].rename(columns={'Close': 'price'})

def prepare_time_features(df):
    """Create time features for Prophet"""
    df = df.reset_index()
    df = df.rename(columns={'Date': 'ds', 'price': 'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)
    return df

if __name__ == "__main__":
    from utils import load_config
    config = load_config()
    
    # Fetch and save data
    price_data = get_stock_data(config['ticker'], config['start_date'], config['end_date'])
    price_data.to_csv('stock_prices.csv')
    
    # Prepare Prophet-compatible data
    prophet_data = prepare_time_features(price_data)
    prophet_data.to_csv('prophet_data.csv', index=False)