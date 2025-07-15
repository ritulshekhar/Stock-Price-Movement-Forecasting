from newsapi import NewsApiClient
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load FinBERT model (financial sentiment analysis)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

def get_news_sentiment(api_key, query, from_date, to_date):
    """Fetch news headlines and compute daily sentiment scores"""
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=query,
                                         from_param=from_date,
                                         to=to_date,
                                         language='en',
                                         sort_by='publishedAt')
    
    # Process articles
    sentiments = []
    for article in all_articles['articles']:
        date = datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d')
        text = article['title'] + ". " + (article['description'] or "")
        
        # Sentiment analysis
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiments.append({
            'date': date,
            'sentiment': probs[0][0].item()  # 0:neutral, 1:positive, 2:negative
        })
    
    # Convert to DataFrame and aggregate by date
    df = pd.DataFrame(sentiments)
    df['date'] = pd.to_datetime(df['date']).dt.date
    daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()
    return daily_sentiment

if __name__ == "__main__":
    from utils import load_config
    config = load_config()
    
    # Get sentiment data
    sentiment_df = get_news_sentiment(
        config['newsapi_key'],
        config['ticker'],
        config['start_date'],
        config['end_date']
    )
    sentiment_df.to_csv('news_sentiment.csv', index=False)