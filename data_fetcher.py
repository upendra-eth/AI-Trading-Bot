import yfinance as yf
import pandas as pd
import feedparser

def fetch_historical_data(symbol: str, period: str = '1mo', interval: str = '1h') -> pd.DataFrame:
    """Fetches OHLCV data for an NSE stock."""
    ticker_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return df
    
    # Drop unnecessary columns
    if 'Dividends' in df.columns:
        df = df.drop(columns=['Dividends'])
    if 'Stock Splits' in df.columns:
        df = df.drop(columns=['Stock Splits'])
        
    return df

def fetch_news_sentiment_raw(symbol: str) -> list:
    """Fetches recent news articles for the stock from a free RSS feed."""
    search_query = symbol.replace('.NS', '')
    url = f"https://news.google.com/rss/search?q={search_query}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    
    news = []
    for entry in feed.entries[:10]:
        news.append({
            'title': entry.title,
            'published': entry.published,
            'link': entry.link
        })
    return news
