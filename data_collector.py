import yfinance as yf
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import requests
import time
from config import Config

class DataCollector:
    def __init__(self):
        self.config = Config()
        self.newsapi = NewsApiClient(api_key=self.config.NEWS_API_KEY)
        
    def get_stock_data(self, symbol=None, period=None):
        """
        Fetch stock data from Yahoo Finance
        """
        symbol = symbol or self.config.STOCK_SYMBOL
        period = period or self.config.PERIOD
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Calculate additional features
            data['Returns'] = data['Close'].pct_change()
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Price_Range'] = data['High'] - data['Low']
            data['Price_Change'] = data['Close'] - data['Open']
            
            print(f"Successfully fetched {len(data)} days of stock data for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def get_news_data(self, query=None, days_back=30):
        """
        Fetch news headlines from NewsAPI
        """
        query = query or f"{self.config.STOCK_SYMBOL} stock"
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Get news articles
            articles = self.newsapi.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            # Process articles
            news_data = []
            for article in articles['articles']:
                news_data.append({
                    'date': article['publishedAt'][:10],
                    'title': article['title'],
                    'description': article['description'] or '',
                    'url': article['url']
                })
            
            df = pd.DataFrame(news_data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
            print(f"Successfully fetched {len(df)} news articles")
            return df
            
        except Exception as e:
            print(f"Error fetching news data: {e}")
            # Return sample data if API fails
            return self._get_sample_news_data()
    
    def _get_sample_news_data(self):
        """
        Generate sample news data for testing
        """
        sample_headlines = [
            "Apple reports strong quarterly earnings",
            "Tech stocks rally amid positive market sentiment",
            "Apple announces new product launch",
            "Market volatility increases due to economic uncertainty",
            "Apple stock reaches new highs",
            "Technology sector shows resilience",
            "Apple CEO discusses future growth plans",
            "Market analysts upgrade Apple stock rating"
        ]
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='D'
        )
        
        news_data = []
        for date in dates:
            # Add 1-3 random headlines per day
            num_headlines = np.random.randint(1, 4)
            for _ in range(num_headlines):
                headline = np.random.choice(sample_headlines)
                news_data.append({
                    'date': date,
                    'title': headline,
                    'description': f"Sample description for {headline}",
                    'url': 'https://example.com'
                })
        
        return pd.DataFrame(news_data)
    
    def save_data(self, data, filename):
        """
        Save data to CSV file
        """
        filepath = f"{self.config.DATA_DIR}/{filename}"
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename):
        """
        Load data from CSV file
        """
        filepath = f"{self.config.DATA_DIR}/{filename}"
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded from {filepath}")
            return data
        except FileNotFoundError:
            print(f"File {filepath} not found")
            return None

if __name__ == "__main__":
    # Create directories
    Config.create_directories()
    
    # Initialize data collector
    collector = DataCollector()
    
    # Collect stock data
    stock_data = collector.get_stock_data()
    if stock_data is not None:
        collector.save_data(stock_data, 'stock_data.csv')
    
    # Collect news data
    news_data = collector.get_news_data()
    if news_data is not None:
        collector.save_data(news_data, 'news_data.csv')