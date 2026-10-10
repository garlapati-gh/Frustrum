import re
import pandas as pd

_PAT = {}

def load_tickers():
    try:
        T = pd.read_csv("data/tickers.csv")
        for _, r in T.iterrows():
            names = [r["name"]] + [a for a in str(r["aliases"]).split("|") if a and a != "nan"]
            body = "|".join(re.escape(n) for n in names)
            _PAT[r["ticker"]] = re.compile(rf"\b({body})\b|\${r['ticker']}\b", re.I)
    except FileNotFoundError:
        pass

def link_tickers(doc) -> list[str]:
    if not _PAT: load_tickers()
    if doc.ticker_hint:
        return [doc.ticker_hint]
    return [t for t, p in _PAT.items() if p.search(doc.text)]
