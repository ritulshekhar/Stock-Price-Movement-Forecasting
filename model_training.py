import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from prophet import Prophet

def create_sequences(data, seq_length):
    """Create time-series sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, config):
    """Build and train LSTM model"""
    model = Sequential([
        LSTM(config['lstm_units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(config['lstm_units']),
        Dropout(0.3),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.1,
        verbose=1
    )
    return model

def prophet_forecast(train_data, periods=30):
    """Generate baseline forecast using Prophet"""
    model = Prophet()
    model.fit(train_data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)

if __name__ == "__main__":
    from utils import load_config
    config = load_config()
    
    # Load and merge data
    prices = pd.read_csv('stock_prices.csv', parse_dates=True, index_col=0)
    sentiment = pd.read_csv('news_sentiment.csv', parse_dates=['date'])
    
    merged = prices.join(sentiment.set_index('date'))
    merged['sentiment'] = merged['sentiment'].fillna(0.5)  # Neutral for missing days
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged)
    
    # Create sequences
    X, y = create_sequences(scaled_data, config['sequence_length'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], shuffle=False)
    
    # Train LSTM model
    lstm_model = train_lstm_model(X_train, y_train, config)
    lstm_model.save('lstm_model.keras')
    
    # Generate Prophet forecast
    prophet_data = pd.read_csv('prophet_data.csv')
    prophet_forecast = prophet_forecast(prophet_data)
    prophet_forecast.to_csv('prophet_forecast.csv', index=False)