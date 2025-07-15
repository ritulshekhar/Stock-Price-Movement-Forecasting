"""
Stock Data Preparation Script
Downloads historical stock data and saves to CSV files
"""
import yfinance as yf
import pandas as pd
import os
import time
import numpy as np

def download_stock_data(ticker, period="max", max_retries=3):
    """Download stock data with retry logic"""
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"Downloading {ticker} data (attempt {attempt+1}/{max_retries})...")
            data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True
            )
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            print(f"Successfully downloaded {len(data)} rows for {ticker}")
            return data
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            attempt += 1
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    raise RuntimeError(f"Failed to download {ticker} after {max_retries} attempts")

def normalize_volume(df):
    """Convert volume to millions for readability"""
    if 'Volume' in df.columns:
        df['Volume (Millions)'] = df['Volume'] / 1_000_000
        df.drop('Volume', axis=1, inplace=True)
    return df

def add_basic_features(df):
    """Add essential features for forecasting"""
    # Moving averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Price changes
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Change'] = df['Close'] - df['Open']
    df['High_Low_Spread'] = df['High'] - df['Low']
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    return df

def clean_data(df):
    """Handle missing values and clean dataset"""
    # Forward fill minor gaps
    df.ffill(inplace=True)
    
    # Drop any remaining missing values
    initial_count = len(df)
    df.dropna(inplace=True)
    final_count = len(df)
    
    print(f"Removed {initial_count - final_count} rows with missing values")
    return df

def main():
    """Main data processing pipeline"""
    tickers = ["MSFT", "AAPL", "GOOG", "AMZN", "META"]
    data_dir = "stock_data"
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    master_df = pd.DataFrame()
    
    for ticker in tickers:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}")
            print(f"{'='*50}")
            
            # Download data
            data = download_stock_data(ticker)
            
            # Basic processing
            data = normalize_volume(data)
            data['Ticker'] = ticker
            
            # Feature engineering
            data = add_basic_features(data)
            
            # Data cleaning
            data = clean_data(data)
            
            # Save individual CSV
            file_path = os.path.join(data_dir, f"{ticker}_historical.csv")
            data.to_csv(file_path)
            print(f"Saved {ticker} data to {file_path}")
            
            # Add to master dataframe
            master_df = pd.concat([master_df, data])
            
            print(f"\nFinished processing {ticker}")
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            print("Skipping to next ticker...\n")
    
    # Save combined data
    master_path = os.path.join(data_dir, "all_stocks_combined.csv")
    master_df.to_csv(master_path)
    print(f"Saved combined data to {master_path}")
    
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETE!")
    print(f"Processed {len(tickers)} stocks")
    print("="*50)

if __name__ == "__main__":
    main()