import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import Config

class FeatureEngineer:
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        
    def add_technical_indicators(self, stock_df):
        """
        Add technical indicators to stock data
        """
        df = stock_df.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_hist'] = ta.trend.macd(df['Close']) - ta.trend.macd_signal(df['Close'])
        
        # Bollinger Bands
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'])
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # Stochastic Oscillator
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Average True Range (ATR)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price position within Bollinger Bands
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Price momentum
        df['Price_momentum_5'] = df['Close'].pct_change(5)
        df['Price_momentum_10'] = df['Close'].pct_change(10)
        df['Price_momentum_20'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        return df
    
    def add_lag_features(self, df, target_col='Close', lags=[1, 2, 3, 5, 10]):
        """
        Add lagged features
        """
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df, target_col='Close', windows=[5, 10, 20]):
        """
        Add rolling statistical features
        """
        for window in windows:
            df[f'{target_col}_roll_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_roll_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_roll_min_{window}'] = df[target_col].rolling(window).min()
            df[f'{target_col}_roll_max_{window}'] = df[target_col].rolling(window).max()
        
        return df
    
    def merge_sentiment_features(self, stock_df, sentiment_df):
        """
        Merge sentiment features with stock data
        """
        # Ensure date columns are datetime
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Merge on date
        merged_df = stock_df.merge(
            sentiment_df, 
            left_on=stock_df['Date'].dt.date, 
            right_on=sentiment_df['date'].dt.date,
            how='left'
        )
        
        # Drop duplicate date column
        merged_df = merged_df.drop(['key_0', 'date'], axis=1, errors='ignore')
        
        # Fill missing sentiment values with neutral values
        sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col]
        for col in sentiment_cols:
            if 'compound' in col:
                merged_df[col] = merged_df[col].fillna(0.0)
            elif 'count' in col:
                merged_df[col] = merged_df[col].fillna(0.0)
            else:
                merged_df[col] = merged_df[col].fillna(0.33)
        
        return merged_df
    
    def create_target_variable(self, df, target_col='Close', prediction_days=1):
        """
        Create target variable for prediction
        """
        # Future price
        df['target'] = df[target_col].shift(-prediction_days)
        
        # Binary classification target (price goes up or down)
        df['target_binary'] = (df['target'] > df[target_col]).astype(int)
        
        # Percentage change target
        df['target_pct_change'] = (df['target'] - df[target_col]) / df[target_col]
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare final feature set
        """
        # Select relevant features
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
            'Stoch_K', 'Stoch_D', 'ATR', 'Volume_ratio',
            'Price_momentum_5', 'Price_momentum_10', 'Price_momentum_20',
            'Volatility_5', 'Volatility_20'
        ]
        
        # Add sentiment features if available
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        feature_cols.extend(sentiment_cols)
        
        # Add lagged features
        lag_cols = [col for col in df.columns if '_lag_' in col]
        feature_cols.extend(lag_cols)
        
        # Add rolling features
        roll_cols = [col for col in df.columns if '_roll_' in col]
        feature_cols.extend(roll_cols)
        
        # Filter existing columns only
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Create feature matrix
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        return X, feature_cols
    
    def create_sequences(self, X, y, sequence_length):
        """
        Create sequences for LSTM training
        """
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def scale_features(self, X_train, X_test, y_train, y_test):
        """
        Scale features for training
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Scale target if it's continuous
        if len(y_train.shape) == 1:
            y_train_scaled = self.price_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = self.price_scaler.transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = y_train
            y_test_scaled = y_test
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

def main():
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Load stock data
    try:
        stock_df = pd.read_csv(f"{Config.DATA_DIR}/stock_data.csv")
        print(f"Loaded stock data: {stock_df.shape}")
    except FileNotFoundError:
        print("Stock data file not found. Please run data_collector.py first.")
        return
    
    # Load sentiment data
    try:
        sentiment_df = pd.read_csv(f"{Config.DATA_DIR}/daily_sentiment.csv")
        print(f"Loaded sentiment data: {sentiment_df.shape}")
    except FileNotFoundError:
        print("Sentiment data file not found. Using stock data only.")
        sentiment_df = pd.DataFrame()
    
    # Add technical indicators
    stock_df = fe.add_technical_indicators(stock_df)
    
    # Add lag features
    stock_df = fe.add_lag_features(stock_df)
    
    # Add rolling features
    stock_df = fe.add_rolling_features(stock_df)
    
    # Merge with sentiment data
    if not sentiment_df.empty:
        stock_df = fe.merge_sentiment_features(stock_df, sentiment_df)
    
    # Create target variable
    stock_df = fe.create_target_variable(stock_df)
    
    # Prepare features
    X, feature_cols = fe.prepare_features(stock_df)
    
    # Save processed data
    stock_df.to_csv(f"{Config.DATA_DIR}/processed_stock_data.csv", index=False)
    
    # Save feature columns
    pd.DataFrame({'feature': feature_cols}).to_csv(f"{Config.DATA_DIR}/feature_columns.csv", index=False)
    
    print("Feature engineering completed!")
    print(f"Final dataset shape: {stock_df.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {feature_cols[:10]}...")

if __name__ == "__main__":
    main()