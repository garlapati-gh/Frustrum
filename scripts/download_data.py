import os
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
from src.ingest.prices import load_prices

def main():
    print("Downloading prices from yfinance...")
    tickers = pd.read_csv("data/tickers.csv")["ticker"].tolist()
    # Pull 60 days of daily data so covariance matrix doesn't crash on empty hourly slices
    df = load_prices(tickers, period="60d", interval="1d")
    df.to_csv("data/prices.csv")
    print(f"Prices saved to data/prices.csv with shape {df.shape}")

    print("Generating sample synthetic news in data/raw/sample_news.jsonl...")
    now = datetime.now(timezone.utc)
    docs = []
    # Generate some positive and negative dummy docs spread over time
    for i, t in enumerate(tickers):
        for j in range(5):
            docs.append({
                "doc_id": f"gdelt_{t}_{i}_{j}",
                "ts": (now - timedelta(hours=i*2 + j)).isoformat(),
                "source": "gdelt",
                "kind": "news",
                "text": f"{t} just announced outstanding quarterly profit beating all estimates by a massive margin.",
                "ticker_hint": t
            })
            docs.append({
                "doc_id": f"tweet_{t}_{i}_{j}",
                "ts": (now - timedelta(hours=i*2 + j - 0.5)).isoformat(),
                "source": "tweets",
                "kind": "social",
                "text": f"Wait wait wait, did {t} CEO just resign? Stock is going to tank rapidly!",
                "ticker_hint": t
            })

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/sample_news.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print("Done generating sample files! Ready to run 'make score'.")

if __name__ == "__main__":
    main()
