"""
Stock Price Forecasting Script
Loads prepared data and builds forecasting models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Configuration
plt.style.use('ggplot')
pd.set_option('display.max_columns', 15)
np.random.seed(42)

def load_data(ticker):
    """Load prepared stock data from CSV"""
    file_path = f"stock_data/{ticker}_historical.csv"
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    print(f"Loaded {len(df)} rows for {ticker}")
    return df

def prepare_for_modeling(df, forecast_days=1):
    """Prepare data for machine learning"""
    # Create target variable (future price)
    df['Target'] = df['Close'].shift(-forecast_days)
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Features and target
    features = df.columns.difference(['Target', 'Ticker'])
    X = df[features]
    y = df['Target']
    
    return X, y

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest model"""
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Performance:")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- R²: {r2:.4f}")
    
    # Feature importance
    importances = model.feature_importances_
    feature_imp = pd.Series(importances, index=X_train.columns)
    print("\nTop 10 Features:")
    print(feature_imp.sort_values(ascending=False).head(10))
    
    return model, predictions

def plot_results(y_test, predictions, ticker):
    """Visualize prediction results"""
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    }, index=y_test.index)
    
    plt.figure(figsize=(14, 7))
    results.plot(title=f'{ticker} Stock Price Predictions')
    plt.ylabel('Price ($)')
    plt.savefig(f'{ticker}_predictions.png', bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.savefig(f'{ticker}_scatter.png', bbox_inches='tight')
    plt.show()

def main():
    """Main forecasting pipeline"""
    ticker = "MSFT"  # Change to analyze different stocks
    
    # Load and prepare data
    df = load_data(ticker)
    X, y = prepare_for_modeling(df)
    
    # Time-based train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nData Split:")
    print(f"- Training: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
    print(f"- Testing: {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model and evaluate
    model, predictions = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Visualize results
    plot_results(y_test, predictions, ticker)
    
    print("\nForecasting pipeline completed successfully!")

if __name__ == "__main__":
    main()