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
            signal_id TEXT PRIMARY KEY, doc_id TEXT, ts TEXT, ticker TEXT, kind TEXT, source TEXT,
            event_type TEXT, sentiment REAL, impact INTEGER, confidence REAL, novelty REAL,
            text TEXT, url TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS weights (ts TEXT, ticker TEXT, weight REAL, z REAL)")
        c.execute("""CREATE TABLE IF NOT EXISTS trades (
            ts TEXT, ticker TEXT, old_weight REAL, new_weight REAL, reason TEXT, turnover REAL, cost_bps REAL, headline TEXT)""")
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

def insert_step(out: dict, top_headline: str = ""):
    ts = out["ts"].isoformat() if hasattr(out["ts"], "isoformat") else str(out["ts"])
    rows = [(ts, t, float(w), float(out["z"][t])) for t, w in out["weights"].items()]
    w_prev = out.get("w_prev")
    reason = out.get("reason", "rebalance")
    turnover = float(out.get("turnover", 0.0))
    cost_bps = float(out.get("cost_bps", 0.0))
    headline = out.get("headline", top_headline)

    trade_rows = []
    if w_prev is not None:
        for t, w in out["weights"].items():
            w_old = float(w_prev[t])
            w_new = float(w)
            if abs(w_new - w_old) > 1e-4:
                trade_rows.append((ts, t, w_old, w_new, reason, turnover, cost_bps, headline))
    if not trade_rows:
        # Fallback summary row if initial or no specific ticker diff
        trade_rows.append((ts, "ALL", 0.0, 0.0, reason, turnover, cost_bps, headline))

    with conn() as c:
        c.executemany("INSERT INTO weights VALUES (?,?,?,?)", rows)
        c.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)", trade_rows)

def set_meta(k, v):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))

def read(sql, params=()):
    with conn() as c:
        return pd.read_sql(sql, c, params=params)
