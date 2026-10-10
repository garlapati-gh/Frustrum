import os
import textwrap

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent(content).strip() + '\n')

create_file('src/common/config.py', '''
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
''')

create_file('src/common/schemas.py', '''
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel

class RawDoc(BaseModel):
    doc_id: str
    ts: datetime
    source: str
    kind: Literal["news", "social"]
    text: str
    url: Optional[str] = None
    domain: Optional[str] = None
    ticker_hint: Optional[str] = None

class Signal(BaseModel):
    signal_id: str
    doc_id: str
    ts: datetime
    ticker: str
    kind: str
    source: str
    text: str
    sentiment: float
    event_type: str
    impact: int
    confidence: float
    novelty: float
    url: Optional[str] = None
''')

create_file('src/common/db.py', '''
import sqlite3
import pandas as pd

DB = "data/frustrum.db"

def conn():
    return sqlite3.connect(DB, check_same_thread=False)

def reset():
    with conn() as c:
        c.execute("DROP TABLE IF EXISTS signals")
        c.execute("DROP TABLE IF EXISTS weights")
        c.execute("DROP TABLE IF EXISTS trades")
        c.execute("DROP TABLE IF EXISTS meta")
    init_db()

def init_db():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY, ts TEXT, ticker TEXT, kind TEXT, source TEXT,
            event_type TEXT, sentiment REAL, impact INTEGER, confidence REAL, novelty REAL,
            text TEXT, url TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS weights (ts TEXT, ticker TEXT, weight REAL, z REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS trades (ts TEXT, reason TEXT, turnover REAL, cost_bps REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_sig_ticker_ts ON signals(ticker, ts)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_w_ts ON weights(ts)")

init_db()

def insert_signals(df: pd.DataFrame):
    if df.empty: return
    d = df.copy()
    d["ts"] = d["ts"].astype(str)
    with conn() as c:
        d.to_sql("signals_tmp", c, if_exists="replace", index=False)
        c.execute("INSERT OR IGNORE INTO signals SELECT * FROM signals_tmp")

def insert_step(out: dict):
    ts = out["ts"].isoformat()
    rows = [(ts, t, float(w), float(out["z"][t])) for t, w in out["weights"].items()]
    with conn() as c:
        c.executemany("INSERT INTO weights VALUES (?,?,?,?)", rows)
        c.execute("INSERT INTO trades VALUES (?,?,?,?)", (ts, out["reason"], out["turnover"], out["cost_bps"]))

def set_meta(k, v):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))

def read(sql, params=()):
    with conn() as c:
        return pd.read_sql(sql, c, params=params)
''')

create_file('src/ingest/gdelt.py', '''
import hashlib
from datetime import datetime, timezone
import requests
from src.common.schemas import RawDoc

API = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_gdelt(query: str, timespan: str = "1d", maxrecords: int = 100) -> list[RawDoc]:
    params = {"query": f"({query}) sourcelang:english", "mode": "artlist", "format": "json", "maxrecords": maxrecords, "timespan": timespan, "sort": "datedesc"}
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    try:
        articles = r.json().get("articles", [])
    except ValueError:
        return []
    docs = []
    for a in articles:
        ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        docs.append(RawDoc(doc_id=hashlib.md5(a["url"].encode()).hexdigest(), ts=ts, source="gdelt", kind="news", text=a["title"], url=a["url"], domain=a.get("domain")))
    return docs
''')

create_file('src/ingest/prices.py', '''
import pandas as pd
import yfinance as yf

def load_prices(tickers, period="2y", interval="1h") -> pd.DataFrame:
    df = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)["Close"]
    df = df.dropna(how="all")
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df
''')

create_file('src/ingest/replay.py', '''
import json, time
from pathlib import Path
from src.common.schemas import RawDoc

def load_docs(paths) -> list[RawDoc]:
    docs = [RawDoc(**json.loads(line)) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    return sorted(docs, key=lambda d: d.ts)

def replay(docs, speed: float = 600.0, instant: bool = False):
    prev = None
    for d in docs:
        if prev is not None and not instant:
            time.sleep(max(0.0, (d.ts - prev).total_seconds() / speed))
        prev = d.ts
        yield d
''')

