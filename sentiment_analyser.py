import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
from tqdm import tqdm
import datetime

load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))

def fetch_headlines(ticker, from_, to_):
    query = f"{ticker}"
    all_articles = newsapi.get_everything(q=query, from_param=from_, to=to_, language='en', sort_by='relevancy')
    return [a['title'] for a in all_articles['articles']]

def analyze_sentiment(headlines):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    results = []
    for text in tqdm(headlines):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]
        results.append({'headline':text, 'neg':probs[0], 'neu':probs[1], 'pos':probs[2]})
    return pd.DataFrame(results)

def main():
    today = datetime.date.today()
    earliest_allowed = datetime.date(2025, 6, 14)
    start = max(today - datetime.timedelta(days=365), earliest_allowed + datetime.timedelta(days=1))
    end = today

    all_data = []
    for ticker in ["MSFT", "AAPL", "GOOG", "AMZN", "META"]:
        print(f"Fetching headlines for {ticker} from {start} to {end}")
        headlines = fetch_headlines(ticker, start.isoformat(), end.isoformat())
        if headlines:
            df = analyze_sentiment(headlines)
            df['ticker'] = ticker
            df['date'] = end
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data)
        final_df.to_csv("data/news_sentiment.csv", index=False)
        print("Sentiment data saved to data/news_sentiment.csv")
    else:
        print("No sentiment data fetched.")

if __name__=="__main__":
    main()
