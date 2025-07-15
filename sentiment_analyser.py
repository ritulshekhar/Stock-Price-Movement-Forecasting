from newsapi import NewsApiClient
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm

# Load FinBERT model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

def get_news_sentiment(api_key, query, from_date, to_date):
    """Fetch news headlines and compute daily sentiment scores with pagination"""
    newsapi = NewsApiClient(api_key=api_key)
    current_date = datetime.strptime(from_date, '%Y-%m-%d')
    end_date = datetime.strptime(to_date, '%Y-%m-%d')
    
    sentiments = []
    date_range = pd.date_range(start=from_date, end=to_date)
    
    print(f"Fetching news for {query} from {from_date} to {to_date}")
    
    # Process by month to handle API limits
    for single_date in tqdm(date_range, desc="Processing days"):
        date_str = single_date.strftime('%Y-%m-%d')
        next_day = (single_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            all_articles = newsapi.get_everything(
                q=query,
                from_param=date_str,
                to=next_day,
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            for article in all_articles['articles']:
                pub_date = datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d')
                text = article['title'] + ". " + (article['description'] or "")
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sentiments.append({
                    'date': pub_date,
                    'sentiment': probs[0][0].item()  # 0:neutral, 1:positive, 2:negative
                })
            
            # Respect API rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing {date_str}: {str(e)}")
            time.sleep(1)

    # Convert to DataFrame and aggregate by date
    df = pd.DataFrame(sentiments)
    if df.empty:
        return pd.DataFrame(columns=['date', 'sentiment'])
    
    df['date'] = pd.to_datetime(df['date']).dt.date
    daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()
    return daily_sentiment

if __name__ == "__main__":
    from utils import load_config
    config = load_config()
    
    sentiment_df = get_news_sentiment(
        config['newsapi_key'],
        config['ticker'],
        config['start_date'],
        config['end_date']
    )
    sentiment_df.to_csv('news_sentiment.csv', index=False)
    print(f"Saved sentiment data with {len(sentiment_df)} days")