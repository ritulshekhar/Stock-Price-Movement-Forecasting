import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import re
from tqdm import tqdm
from config import Config

class SentimentAnalyzer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load FinBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.SENTIMENT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.SENTIMENT_MODEL)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    def preprocess_text(self, text):
        """
        Clean and preprocess text for sentiment analysis
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length to avoid tokenization issues
        return text[:512]
    
    def get_sentiment_score(self, text):
        """
        Get sentiment score for a single text
        """
        if not text or text.strip() == '':
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33, 'compound': 0.0}
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()[0]
            
            # Create sentiment scores
            sentiment_scores = {
                'negative': float(predictions[0]),
                'neutral': float(predictions[1]),
                'positive': float(predictions[2])
            }
            
            # Calculate compound score
            compound = sentiment_scores['positive'] - sentiment_scores['negative']
            sentiment_scores['compound'] = compound
            
            return sentiment_scores
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33, 'compound': 0.0}
    
    def analyze_news_sentiment(self, news_df):
        """
        Analyze sentiment for news dataframe
        """
        if news_df.empty:
            return news_df
        
        print("Analyzing news sentiment...")
        
        # Combine title and description
        news_df['combined_text'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
        
        # Initialize sentiment columns
        sentiment_columns = ['negative', 'neutral', 'positive', 'compound']
        for col in sentiment_columns:
            news_df[f'sentiment_{col}'] = 0.0
        
        # Process each news article
        for idx, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Processing news"):
            sentiment_scores = self.get_sentiment_score(row['combined_text'])
            
            for col in sentiment_columns:
                news_df.loc[idx, f'sentiment_{col}'] = sentiment_scores[col]
        
        return news_df
    
    def aggregate_daily_sentiment(self, news_df):
        """
        Aggregate sentiment scores by date
        """
        if news_df.empty:
            return pd.DataFrame()
        
        # Ensure date column is datetime
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Group by date and calculate aggregated sentiment
        daily_sentiment = news_df.groupby(news_df['date'].dt.date).agg({
            'sentiment_negative': ['mean', 'std', 'count'],
            'sentiment_neutral': ['mean', 'std'],
            'sentiment_positive': ['mean', 'std'],
            'sentiment_compound': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = ['date'] + [f'{col[0]}_{col[1]}' for col in daily_sentiment.columns[1:]]
        
        # Fill NaN values
        daily_sentiment = daily_sentiment.fillna(0)
        
        # Add additional sentiment features
        daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_compound_mean'].rolling(window=3).mean()
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_compound_std'].rolling(window=7).mean()
        
        return daily_sentiment
    
    def get_sentiment_signals(self, daily_sentiment):
        """
        Generate trading signals based on sentiment
        """
        if daily_sentiment.empty:
            return daily_sentiment
        
        # Bullish signal: high positive sentiment
        daily_sentiment['bullish_signal'] = (
            (daily_sentiment['sentiment_positive_mean'] > 0.6) &
            (daily_sentiment['sentiment_compound_mean'] > 0.3)
        ).astype(int)
        
        # Bearish signal: high negative sentiment
        daily_sentiment['bearish_signal'] = (
            (daily_sentiment['sentiment_negative_mean'] > 0.6) &
            (daily_sentiment['sentiment_compound_mean'] < -0.3)
        ).astype(int)
        
        # Neutral signal
        daily_sentiment['neutral_signal'] = (
            (daily_sentiment['sentiment_neutral_mean'] > 0.5) &
            (abs(daily_sentiment['sentiment_compound_mean']) < 0.1)
        ).astype(int)
        
        return daily_sentiment

def main():
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load news data
    try:
        news_df = pd.read_csv(f"{Config.DATA_DIR}/news_data.csv")
        print(f"Loaded {len(news_df)} news articles")
    except FileNotFoundError:
        print("News data file not found. Please run data_collector.py first.")
        return
    
    # Analyze sentiment
    news_with_sentiment = analyzer.analyze_news_sentiment(news_df)
    
    # Aggregate daily sentiment
    daily_sentiment = analyzer.aggregate_daily_sentiment(news_with_sentiment)
    
    # Generate sentiment signals
    daily_sentiment = analyzer.get_sentiment_signals(daily_sentiment)
    
    # Save results
    news_with_sentiment.to_csv(f"{Config.DATA_DIR}/news_with_sentiment.csv", index=False)
    daily_sentiment.to_csv(f"{Config.DATA_DIR}/daily_sentiment.csv", index=False)
    
    print("Sentiment analysis completed!")
    print(f"Daily sentiment shape: {daily_sentiment.shape}")
    print(f"Average compound sentiment: {daily_sentiment['sentiment_compound_mean'].mean():.3f}")

if __name__ == "__main__":
    main()