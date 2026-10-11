import threading
import copy
import pandas as pd
from fastapi import FastAPI
from src.common import db
from src.common.config import load_config
from src.runner import run_replay

app = FastAPI(title="Frustrum")
cfg = load_config()
_stop = threading.Event()

@app.get("/health")
def health():
    try:
        row = db.read("SELECT value FROM meta WHERE key='sim_time'")
        return {"ok": True, "sim_time": row["value"].iloc[0] if len(row) else None}
    except Exception:
        return {"ok": False}

@app.get("/signals")
def signals(ticker: str | None = None, since: str | None = None, limit: int = 100):
    q, p = "SELECT * FROM signals WHERE 1=1", []
    if ticker: q += " AND ticker=?"; p.append(ticker)
    if since: q += " AND ts>=?"; p.append(since)
    q += " ORDER BY ts DESC LIMIT ?"; p.append(limit)
    return db.read(q, tuple(p)).to_dict("records")

@app.get("/weights/latest")
def weights_latest():
    return db.read("SELECT ticker, weight, z FROM weights WHERE ts=(SELECT MAX(ts) FROM weights)").to_dict("records")

@app.get("/weights/history")
def weights_history(since: str | None = None):
    q = "SELECT * FROM weights" + (" WHERE ts>=?" if since else " ") + " ORDER BY ts"
    return db.read(q, (since,) if since else ()).to_dict("records")

@app.get("/explain/{ticker}")
def explain_ticker(ticker: str, n: int = 5):
    return db.read("SELECT * FROM signals WHERE ticker=? ORDER BY ts DESC LIMIT ?", (ticker, n)).to_dict("records")

@app.post("/replay/start")
def replay_start(speed: float = 600.0, start: str | None = None, end: str | None = None,
                 lambda_tilt: float | None = None, turnover_cap: float | None = None,
                 cost_bps: float | None = None):
    _stop.set(); _stop.clear(); db.reset()
    c = copy.deepcopy(cfg)
    for k, v in {"lambda_tilt": lambda_tilt, "turnover_cap": turnover_cap, "cost_bps": cost_bps}.items():
        if v is not None: c["rebalance"][k] = v
    s = pd.Timestamp(start, tz="UTC") if start else None
    e = pd.Timestamp(end, tz="UTC") if end else None
    threading.Thread(target=run_replay, args=(c, speed, _stop, s, e), daemon=True).start()
    return {"started": True, "speed": speed}

@app.post("/replay/stop")
def replay_stop():
    _stop.set()
    return {"stopped": True}
