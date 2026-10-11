"""
Generate synthetic demo data spanning 2 years of daily price history.
Signals are deliberately spread across the full date range so the backtest
engine has live signals at every historical timestamp. This is synthetic data,
clearly labelled as such.
"""
import os
import json
import random
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
from src.ingest.prices import load_prices


TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "WMT", "KO", "BA", "DIS"]

# Templates by event type – (positive text, negative text)
TEMPLATES = {
    "earnings": [
        "{t} reports blowout quarterly earnings, revenue surges {pct}% above estimates.",
        "{t} misses earnings estimates badly; revenue falls short by {pct}% this quarter.",
        "{t} raises full-year guidance after record EPS of ${eps}.",
        "{t} cuts full-year guidance citing rising costs and weak demand.",
    ],
    "credit": [
        "Moody's upgrades {t} credit rating citing strong balance sheet.",
        "S&P cuts {t} debt rating to junk, cites unsustainable leverage.",
        "{t} secures new $10 billion credit facility at favourable rates.",
        "{t} faces potential default risk as debt maturities pile up.",
    ],
    "ma": [
        "{t} to acquire rival in landmark ${val}B all-cash deal.",
        "{t} merger talks collapse; deal called off amid regulatory concerns.",
        "{t} completes strategic acquisition of key tech startup.",
        "{t} walks away from takeover bid, citing valuation disagreements.",
    ],
    "regulatory": [
        "Regulators clear {t} of antitrust allegations, stock surges.",
        "DOJ launches antitrust probe into {t} business practices.",
        "{t} wins landmark patent ruling, protecting key product line.",
        "SEC fines {t} ${val}M over disclosure violations.",
    ],
    "geopolitical": [
        "New trade agreement boosts {t} international revenue outlook.",
        "US sanctions disrupt {t} supply chain; exports blocked.",
        "{t} benefits from de-escalation in key overseas markets.",
        "Tariffs imposed on {t} products; significant revenue at risk.",
    ],
    "management": [
        "{t} names proven industry veteran as new CEO.",
        "{t} CEO unexpectedly resigns amid board disagreements.",
        "{t} board approves major share buyback and dividend raise.",
        "{t} key executive team departs; leadership vacuum feared.",
    ],
    "product": [
        "{t} unveils breakthrough product with revolutionary technology.",
        "{t} issues massive product recall affecting millions of units.",
        "{t} launches next-generation platform, exceeding analyst expectations.",
        "{t} delays flagship product launch citing quality control issues.",
    ],
    "macro": [
        "Fed signals rate pause; financial sector broadly rallies.",
        "Fed raises rates aggressively; growth stocks including {t} sell off.",
        "Strong GDP data fuels risk-on rally across equity markets.",
        "Inflation spikes to 40-year high; central banks signal tightening.",
    ],
}

# Sentiment direction per template index (0,2 positive; 1,3 negative)
SENTIMENT_SIGN = [1, -1, 1, -1]
SOURCE_POOL = ["gdelt", "newsapi", "tweets", "gdelt", "newsapi"]
KIND_POOL   = ["news", "news", "social", "news", "news"]


def make_doc(doc_id, ts, ticker, etype, template_idx):
    tmpl = TEMPLATES[etype][template_idx % len(TEMPLATES[etype])]
    text = tmpl.format(
        t=ticker,
        pct=random.randint(3, 25),
        eps=round(random.uniform(1.5, 8.0), 2),
        val=random.randint(2, 50),
    )
    src_idx = random.randint(0, len(SOURCE_POOL) - 1)
    return {
        "doc_id": doc_id,
        "ts": ts.isoformat(),
        "source": SOURCE_POOL[src_idx],
        "kind": KIND_POOL[src_idx],
        "text": text,
        "ticker_hint": ticker,
        "event_hint": etype,
    }


def main():
    print("Downloading 2 years of daily prices from yfinance...")
    df = load_prices(TICKERS, period="2y", interval="1d")
    df.to_csv("data/prices.csv")
    print(f"Prices saved: {df.shape[0]} days × {df.shape[1]} tickers")

    # Save a tiny sample for the repo
    df.tail(30).to_csv("data/prices_sample.csv")

    # Date range from prices
    start_dt = df.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
    end_dt   = df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    date_range_days = (end_dt - start_dt).days
    print(f"Price range: {start_dt.date()} to {end_dt.date()} ({date_range_days} days)")

    print("Generating synthetic news events across 2-year price range...")
    random.seed(42)
    np.random.seed(42)

    docs = []
    etypes = list(TEMPLATES.keys())
    doc_counter = 0

    # ~5 events per ticker per event type → well-distributed coverage
    events_per_ticker_per_type = 5
    for ticker in TICKERS:
        for etype in etypes:
            for k in range(events_per_ticker_per_type):
                offset_days = random.randint(0, date_range_days - 1)
                offset_hours = random.randint(8, 20)
                ts = start_dt + timedelta(days=offset_days, hours=offset_hours)
                tmpl_idx = (k * 2) % len(TEMPLATES[etype])  # alternate pos/neg
                docs.append(make_doc(f"syn_{doc_counter}", ts, ticker, etype, tmpl_idx))
                doc_counter += 1

    # Plant ONE deliberate circuit-breaker event on BA
    # (fraud headline — impact 10, very negative)
    cb_ts = start_dt + timedelta(days=int(date_range_days * 0.7), hours=14)
    docs.append({
        "doc_id": "cb_ba_fraud",
        "ts": cb_ts.isoformat(),
        "source": "newsapi",
        "kind": "news",
        "text": "Boeing faces fraud probe: SEC investigates accounting irregularities worth billions.",
        "ticker_hint": "BA",
        "event_hint": "regulatory",
    })

    # Also add some broader macro events (ticker="*")
    for k in range(8):
        offset_days = random.randint(0, date_range_days - 1)
        ts = start_dt + timedelta(days=offset_days, hours=15)
        macro_text = random.choice([
            "Federal Reserve raises interest rates by 75 bps in surprise move.",
            "Fed signals pause in rate hikes; inflation shows signs of cooling.",
            "US GDP growth beats expectations in Q3, risk appetite rises.",
            "Banking crisis fears mount as regional lenders face deposit runs.",
        ])
        docs.append({
            "doc_id": f"macro_{k}",
            "ts": ts.isoformat(),
            "source": "newsapi",
            "kind": "news",
            "text": macro_text,
            "ticker_hint": "*",
            "event_hint": "macro",
        })

    # Sort by timestamp
    docs.sort(key=lambda d: d["ts"])

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/sample_news.jsonl"
    with open(out_path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    # Also write a committed sample (first 200 rows)
    sample_path = "data/sample_news.jsonl"
    with open(sample_path, "w") as f:
        for d in docs[:200]:
            f.write(json.dumps(d) + "\n")

    print(f"Generated {len(docs)} synthetic events -> {out_path}")
    print(f"Sample (200 rows) -> {sample_path}")
    print("Event type breakdown:")
    from collections import Counter
    counts = Counter(d.get("event_hint", "?") for d in docs)
    for et, c in sorted(counts.items()):
        print(f"  {et:15s}: {c}")
    print("\nDone. Run 'python -m scripts.precompute_scores' next.")


if __name__ == "__main__":
    main()
