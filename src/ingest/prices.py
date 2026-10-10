import pandas as pd
import yfinance as yf

def load_prices(tickers, period="2y", interval="1h") -> pd.DataFrame:
    df = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)["Close"]
    df = df.dropna(how="all")
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df
