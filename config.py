import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_newsapi_key_here')
    
    # Stock Configuration
    STOCK_SYMBOL = 'AAPL'
    PERIOD = '2y'  # 2 years of historical data
    
    # Model Configuration
    SEQUENCE_LENGTH = 60  # Days to look back for LSTM
    PREDICTION_DAYS = 30  # Days to predict ahead
    
    # Sentiment Analysis
    SENTIMENT_MODEL = 'ProsusAI/finbert'
    NEWS_SEARCH_TERMS = ['stock market', 'economy', 'finance', 'earnings']
    
    # Feature Engineering
    TECHNICAL_INDICATORS = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
    
    # Model Training
    TRAIN_SPLIT = 0.8
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Directories
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    PLOTS_DIR = 'plots'
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.DATA_DIR, cls.MODEL_DIR, cls.PLOTS_DIR]:
            os.makedirs(directory, exist_ok=True)