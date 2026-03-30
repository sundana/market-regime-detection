import yfinance as yf
import pandas as pd
from pathlib import Path

def get_cached_yfinance_data(ticker, period='6y', interval='1d'):
    """
    Downloads data from yfinance or loads it from a local Parquet cache.
    """
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)

    file_name = f"{ticker.replace('=','_')}_{period}_{interval}.parquet"
    file_path = cache_dir / file_name
    if file_path.exists():
        print(f"Loading {ticker} from local cache: {file_path}")
        return pd.read_parquet(file_path)
    
    print(f"Downloading {ticker} from yfinance...")
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.to_parquet(file_path)
    print(f"Data saved to cache {file_path}")

    return df


if __name__ == "__main__":
    df = get_cached_yfinance_data('XAUUSD=X')
    print(df.tail())