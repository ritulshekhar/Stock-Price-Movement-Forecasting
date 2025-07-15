import pandas as pd
import matplotlib.pyplot as plt
import os
from accuracy_metrics import accuracy_predictions
import numpy as np

# Define technical indicators
def add_technical_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load and clean stock price data
stock_file = 'data/MSFT_stock_prices.csv'
if not os.path.exists(stock_file):
    raise FileNotFoundError(f"File not found: {stock_file}")

# Skip first 2 metadata rows
df = pd.read_csv(stock_file, skiprows=2)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df.dropna(subset=['Date'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df.set_index('Date', inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Add technical indicators
df = add_technical_indicators(df)

# Plotting moving averages
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', color='black')
plt.plot(df['MA_10'], label='MA 10 Days', color='blue')
plt.plot(df['MA_50'], label='MA 50 Days', color='red')
plt.title('MSFT Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Save with indicators
df.to_csv('data/MSFT_stock_with_indicators.csv')
print("✅ Technical indicators calculated and chart displayed.")
