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

from pydantic import BaseModel
from src.nlp.pipeline import process
from src.common.schemas import RawDoc
from datetime import datetime, timezone

class ScoreRequest(BaseModel):
    text: str
    ticker_hint: str | None = None
    kind: str = "news"
    source: str = "synthetic_news"

@app.get("/health")
def health():
    try:
        row = db.read("SELECT value FROM meta WHERE key='sim_time'")
        err_row = db.read("SELECT value FROM meta WHERE key='error'")
        sim_time = row["value"].iloc[0] if len(row) else None
        err = err_row["value"].iloc[0] if len(err_row) else None
        return {"ok": err is None, "sim_time": sim_time, "error": err}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/score")
def score_text(req: ScoreRequest):
    doc = RawDoc(
        doc_id="interactive_demo",
        ts=datetime.now(timezone.utc),
        source=req.source,
        kind=req.kind if req.kind in ["news", "social"] else "news",
        text=req.text,
        ticker_hint=req.ticker_hint
    )
    res = process([doc])
    if res:
        return res[0].model_dump()
    return {"error": "Failed to score headline"}

@app.get("/trades")
def trades(limit: int = 50):
    q = "SELECT ts, ticker, old_weight, new_weight, reason, turnover, cost_bps, headline FROM trades WHERE ticker != 'ALL' ORDER BY ts DESC LIMIT ?"
    res = db.read(q, (limit,)).to_dict("records")
    if not res:
        # fallback to all rows
        res = db.read("SELECT * FROM trades ORDER BY ts DESC LIMIT ?", (limit,)).to_dict("records")
    return res

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
