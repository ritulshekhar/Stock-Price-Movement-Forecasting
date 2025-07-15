import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import Config
import ta

class FeatureEngineer:
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()

    def add_technical_indicators(self, df):
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['BB_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        return df

    def create_target(self, df, days=1):
        df['Target'] = df['Close'].shift(-days)
        return df.dropna()

    def merge_sentiment(self, df, sentiment_df):
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        df['Date'] = pd.to_datetime(df['Date'])
        merged = pd.merge(df, sentiment_df, left_on=df['Date'].dt.date, right_on=sentiment_df['date'].dt.date, how='left')
        merged = merged.drop(['key_0', 'date'], axis=1, errors='ignore')
        sentiment_cols = [col for col in merged.columns if 'sentiment' in col]
        for col in sentiment_cols:
            merged[col] = merged[col].fillna(0.33)
        return merged

    def scale_data(self, X_train, X_test, y_train, y_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        y_train_scaled = self.price_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.price_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
