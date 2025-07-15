import yfinance as yf
import pandas as pd
import os
import time
import numpy as np
import talib

def download_stock_data(ticker, period="max", max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"Downloading {ticker} data (attempt {attempt+1}/{max_retries})...")
            
            # Explicitly set auto_adjust to avoid warning
            data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True  # Set explicitly to avoid warning
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
                
            # Add ticker symbol as a column
            data['Ticker'] = ticker
                
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
    """Convert volume to millions for easier reading"""
    if 'Volume' in df.columns:
        df['Volume (Millions)'] = df['Volume'] / 1_000_000
        df.drop('Volume', axis=1, inplace=True)
    return df

def add_features(df, ticker):
    """Add technical indicators and feature engineering"""
    print(f"Adding features for {ticker}...")
    
    # 1. Moving Averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # 2. Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 3. Additional Features
    df['Price_Change'] = df['Close'] - df['Open']
    df['High_Low_Spread'] = df['High'] - df['Low']
    
    # 4. Technical Indicators (using TA-Lib)
    df['RSI'] = talib.RSI(df['Close'])
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
    df['Stochastic_K'], df['Stochastic_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    
    # 5. Volatility Measures
    df['Volatility'] = df['Daily_Return'].rolling(window=30).std()
    
    return df

def validate_and_clean(df, ticker):
    """Check for missing values and clean data"""
    print(f"Validating data for {ticker}...")
    
    # 1. Missing value check
    print(f"Missing values in {ticker}:")
    print(df.isnull().sum())
    
    # 2. Fill minor gaps (forward fill)
    df.ffill(inplace=True)
    
    # 3. Drop any remaining rows with missing values
    initial_count = len(df)
    df.dropna(inplace=True)
    final_count = len(df)
    
    print(f"Dropped {initial_count - final_count} rows with missing values")
    
    return df

def main():
    tickers = ["MSFT", "AAPL", "GOOG", "AMZN", "META"]
    data_dir = "stock_data"
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a master dataframe
    master_df = pd.DataFrame()
    
    for ticker in tickers:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}")
            print(f"{'='*50}")
            
            # 1. Download data
            data = download_stock_data(ticker)
            
            # 2. Normalize volume
            data = normalize_volume(data)
            
            # 3. Add ticker column
            data['Ticker'] = ticker
            
            # ===== NEW: FEATURE ENGINEERING =====
            # 4. Add moving averages
            print("Adding moving averages...")
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['MA_200'] = data['Close'].rolling(window=200).mean()
            
            # 5. Add daily returns
            print("Adding daily returns...")
            data['Daily_Return'] = data['Close'].pct_change()
            # ===== END FEATURE ENGINEERING =====
            
            # ===== NEW: DATA VALIDATION =====
            # 6. Check for missing values
            print("\nMissing values before cleaning:")
            print(data.isnull().sum())
            
            # 7. Fill minor gaps (forward fill)
            print("\nFilling missing values...")
            data.ffill(inplace=True)
            
            # 8. Drop any remaining NaN values
            initial_rows = len(data)
            data.dropna(inplace=True)
            final_rows = len(data)
            print(f"Dropped {initial_rows - final_rows} rows with missing values")
            # ===== END DATA VALIDATION =====
            
            # 9. Print sample
            print(f"\nRecent 5 rows for {ticker} (with new features):")
            print(data.tail())  # Show recent data which has all features
            
            # 10. Save individual CSV
            file_path = os.path.join(data_dir, f"{ticker}_historical.csv")
            data.to_csv(file_path)
            print(f"Saved {ticker} data to {file_path}")
            
            # 11. Add to master dataframe
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