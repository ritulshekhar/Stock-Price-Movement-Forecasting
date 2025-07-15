import yfinance as yf
import os

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

tickers = ["MSFT", "AAPL", "GOOG", "AMZN", "META"]
start_date = "2024-07-16"  # adjust to your needed start date
end_date = "2025-07-16"    # adjust to your needed end date

for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        # Save to CSV
        data.to_csv(f"data/{ticker}_stock_prices.csv")
        print(f"Saved {ticker}_stock_prices.csv")
    else:
        print(f"No data found for {ticker}")

print("Download complete.")
