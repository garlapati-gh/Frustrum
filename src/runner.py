import time
import pandas as pd
from src.common import db
from src.rebalance.engine import Rebalancer

def build_rebalancer(cfg, prices):
    tickers = list(prices.columns)
    try:
        T = pd.read_csv("data/tickers.csv")
        sector_of = dict(zip(T["ticker"], T["sector"]))
    except Exception:
        sector_of = {t: "Technology" for t in tickers}
    try:
        S = pd.read_csv("data/sector_sensitivity.csv", index_col=0)
        sector_beta = S.to_dict()
    except Exception:
        sector_beta = {"macro": {}, "geopolitical": {}}
    bench = pd.Series(1 / len(tickers), index=tickers)
    return Rebalancer(cfg, tickers, sector_of, sector_beta, bench)

def tick_batches(sig: pd.DataFrame, step_min: int = 5):
    step = pd.Timedelta(minutes=step_min)
    t, end = sig["ts"].min().floor(f"{step_min}min"), sig["ts"].max()
    while t <= end:
        yield t, sig[(sig["ts"] > t - step) & (sig["ts"] <= t)]
        t += step

def run_replay(cfg, speed, stop, start=None, end=None):
    import traceback
    try:
        try:
            sig = pd.read_json("data/signals.jsonl", lines=True)
            sig["ts"] = pd.to_datetime(sig["ts"], utc=True)
        except Exception as e:
            sig = pd.DataFrame(columns=["ts", "ticker", "kind", "source", "event_type", "sentiment", "impact", "confidence", "novelty", "text", "url"])
            
        if start: 
            sig = sig[(sig["ts"] >= start) & (sig["ts"] <= end)]   # demo window: 2 to 3 days
            
        try:
            prices = pd.read_csv("data/prices.csv", index_col=0)
            prices.index = pd.to_datetime(prices.index, utc=True)
        except Exception:
            # Fallback to dummy prices if missing
            dates = pd.date_range(start="2022-01-01", periods=100, freq="1H", tz="UTC")
            prices = pd.DataFrame(100.0, index=dates, columns=cfg["universe"])
            
        rb = build_rebalancer(cfg, prices)            # loads tickers.csv, sensitivity csv, equal-weight benchmark
        
        if not sig.empty:
            initial_ts = sig["ts"].min().floor("5min")
        else:
            initial_ts = pd.Timestamp.utcnow()
            
        db.insert_step({
            "ts": initial_ts,
            "weights": rb.w,
            "z": rb.z_last,
            "turnover": 0.0,
            "cost_bps": 0.0,
            "reason": "initial"
        })
        
        for now, new in tick_batches(sig):
            if stop.is_set(): break
            db.insert_signals(new)
            if new.empty and now.minute % 30:         # quiet tick: only check every 30 minutes
                continue
            out = rb.step(now, sig[sig["ts"] <= now], prices, new)
            if out: db.insert_step(out)
            db.set_meta("sim_time", now.isoformat())
            time.sleep(5 * 60 / speed)
    except Exception as e:
        with open("replay_error.log", "w") as f:
            f.write(traceback.format_exc())
