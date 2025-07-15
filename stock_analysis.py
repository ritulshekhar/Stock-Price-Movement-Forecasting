# stock_analysis.py

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 2. Load data
msft = pd.read_csv("stock_data/MSFT_historical.csv", index_col="Date", parse_dates=True)
aapl = pd.read_csv("stock_data/AAPL_historical.csv", index_col="Date", parse_dates=True)

# 3. Feature engineering (add technical indicators)
def add_technical_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    return df

msft = add_technical_indicators(msft)
aapl = add_technical_indicators(aapl)

# 4. Handle missing values (from technical indicators)
msft.dropna(inplace=True)
aapl.dropna(inplace=True)

# ===== ADD PREPARE_FOR_MODELING FUNCTION HERE =====
def prepare_for_modeling(df, forecast_days=1):
    # Create target variable (future price)
    df['Target'] = df['Close'].shift(-forecast_days)
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Features and target
    features = df.columns.difference(['Target', 'Ticker'])
    X = df[features]
    y = df['Target']
    
    return X, y
# ===== END FUNCTION =====

# 5. Prepare data for modeling
X_msft, y_msft = prepare_for_modeling(msft.copy())

# 6. Time-based split
split_idx = int(len(X_msft) * 0.8)
X_train, X_test = X_msft[:split_idx], X_msft[split_idx:]
y_train, y_test = y_msft[:split_idx], y_msft[split_idx:]

# 7. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate model
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse:.2f}")

# 9. Visualize results
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted', alpha=0.7)
plt.title('Microsoft Stock Price Prediction')
plt.legend()
plt.show()