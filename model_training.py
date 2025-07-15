import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import load_config
import os

config = load_config()

def evaluate_predictions(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("\n📊 Evaluation Metrics for LSTM Model:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Load stock prices
stock_file = 'data/MSFT_stock_prices.csv'
if not os.path.exists(stock_file):
    raise FileNotFoundError(f"CSV not found: {stock_file}. Run the stock downloader script first.")

# Read the file and skip extra metadata rows
prices = pd.read_csv(stock_file, skiprows=2)
prices.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
prices = prices.dropna(subset=['Date'])
prices['Date'] = pd.to_datetime(prices['Date'], errors='coerce')
prices = prices.dropna(subset=['Date'])
prices.set_index('Date', inplace=True)
prices = prices.apply(pd.to_numeric, errors='coerce')
prices = prices.dropna()
prices['date'] = prices.index.date

if 'Close' not in prices.columns:
    raise KeyError("'Close' column missing from prices data.")

print("📊 Prices date range:", prices.index.min(), "to", prices.index.max())

# Load sentiment data
sentiment_file = 'data/news_sentiment.csv'
if not os.path.exists(sentiment_file):
    raise FileNotFoundError(f"Sentiment CSV not found: {sentiment_file}. Run sentiment_analyser.py first.")

sentiment = pd.read_csv(sentiment_file, parse_dates=['date'])
sentiment['date_only'] = sentiment['date'].dt.date
sentiment['sentiment'] = sentiment['pos'] - sentiment['neg']

print("🗞️ Sentiment date range:", sentiment['date_only'].min(), "to", sentiment['date_only'].max())

# Merge without trimming
merged = prices.copy()
merged = merged.merge(sentiment[['date_only', 'sentiment']], left_on='date', right_on='date_only', how='left')
merged['sentiment'] = merged['sentiment'].fillna(0.0)

print(f"Merged data shape: {merged.shape}")
print("Sample merged data:")
print(merged.head())

# Feature scaling
numeric_cols = merged.select_dtypes(include=[np.number]).columns
print(f"Numeric columns used for scaling: {numeric_cols.tolist()}")

scaler = MinMaxScaler()
if merged[numeric_cols].shape[0] == 0:
    raise ValueError("Merged data is empty after processing. Check for missing or mismatched dates.")
scaled_data = scaler.fit_transform(merged[numeric_cols])

# Sequence creation
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, list(numeric_cols).index('Close')])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, config['sequence_length'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], shuffle=False)

# LSTM model
model = Sequential([
    LSTM(config['lstm_units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(config['lstm_units']),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=0.1, verbose=1)

# Predict on test data
y_pred = model.predict(X_test).flatten()

# Evaluate predictions
metrics = evaluate_predictions(y_test, y_pred)

os.makedirs('models', exist_ok=True)
model.save('models/lstm_model.keras')

# Prophet model
prophet_df = prices.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=config['sequence_length'])
forecast = prophet_model.predict(future)

os.makedirs('data', exist_ok=True)
forecast[['ds', 'yhat']].tail(config['sequence_length']).to_csv('data/prophet_forecast.csv', index=False)

print("\n✅ LSTM and Prophet models trained and saved successfully.")
