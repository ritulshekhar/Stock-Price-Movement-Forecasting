import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from utils import load_config

def get_stock_data(ticker, start_date, end_date, max_retries=3):
    """Fetch historical stock data with retries"""
    for attempt in range(max_retries):
        try:
            print(f"Downloading {ticker} data (attempt {attempt+1})...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                return data[['Close']].rename(columns={'Close': 'price'})
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
    raise Exception(f"Failed to download {ticker} after {max_retries} attempts")

def prepare_time_features(df):
    """Create time features for Prophet"""
    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Reset index to work with Prophet
    df = df.reset_index()
    df = df.rename(columns={'Date': 'ds', 'price': 'y'})
    
    # Convert to timezone-naive datetime
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    return df

if __name__ == "__main__":
    config = load_config()
    
    # Fetch and save data
    try:
        price_data = get_stock_data(config['ticker'], config['start_date'], config['end_date'])
        print("Successfully downloaded stock data:")
        print(price_data.head())
        
        price_data.to_csv('stock_prices.csv')
        print("Saved stock prices to stock_prices.csv")
        
        # Prepare Prophet-compatible data
        prophet_data = prepare_time_features(price_data)
        prophet_data.to_csv('prophet_data.csv', index=False)
        print("Saved Prophet-formatted data to prophet_data.csv")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        print("Possible solutions:")
        print("1. Try a different ticker (e.g., 'MSFT')")
        print("2. Check your internet connection")
        print("3. Try again later - Yahoo Finance might be temporarily unavailable")