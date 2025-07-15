import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_stock_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ CSV not found at {filepath}. Please generate it first.")

    # Skip first two header rows
    df = pd.read_csv(filepath, skiprows=2)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)

    # Ensure all data is numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    if 'Close' not in df.columns:
        raise KeyError("❌ 'Close' column missing from loaded data. Check if CSV headers are correct.")

    print(f"✅ Loaded {len(df)} rows for MSFT")
    return df

def prepare_for_modeling(df, forecast_days=5):
    df['Target'] = df['Close'].shift(-forecast_days)

    df = df.dropna()
    features = df[['Open', 'High', 'Low', 'Volume', 'Close']]
    target = df['Target']

    return train_test_split(features, target, test_size=0.2, shuffle=False)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"📉 Mean Squared Error: {mse:.4f}")

    return model, predictions

def plot_results(y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title("Stock Price Forecasting")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    filepath = "data/MSFT_stock_prices.csv"
    df = load_stock_data(filepath)
    X_train, X_test, y_train, y_test = prepare_for_modeling(df)
    model, predictions = train_and_evaluate(X_train, X_test, y_train, y_test)
    plot_results(y_test, predictions)

if __name__ == "__main__":
    main()
