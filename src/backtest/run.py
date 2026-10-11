import copy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.common.config import load_config
from src.runner import build_rebalancer
from src.backtest.metrics import perf

SCENARIOS = {
    "Equal-weight, hold":        ("bench_hold", {}),
    "Equal-weight, daily rebal": ("bench_daily", {}),
    "Naive sentiment tilt":      ("naive", {}),
    "Frustrum (full)":          ("riskpulse", {}),
    "Frustrum - no decay":      ("riskpulse", {"ablation": {"decay": False}}),
    "Frustrum - no dedup":      ("riskpulse", {"ablation": {"dedup": False}}),
    "Frustrum - no contagion":  ("riskpulse", {"ablation": {"contagion": False}}),
    "Frustrum - no cost ctrl":  ("riskpulse", {"ablation": {"cost_control": False}}),
}

def override(cfg, patch):
    c = copy.deepcopy(cfg)
    for k, v in patch.items():
        c[k].update(v)
    return c

def naive_target(b, seen, now, tickers):
    recent = seen[seen["ts"] > now - pd.Timedelta(hours=24)]
    s = recent.groupby("ticker")["sentiment"].mean().reindex(tickers, fill_value=0.0)
    w = b * (1 + 0.5 * s)                    # sentiment only: no decay, no confidence, no costs
    return w / w.sum()

def run_strategy(name, cfg, sig, px):
    tickers = list(px.columns)
    b = pd.Series(1 / len(tickers), index=tickers)
    bps = cfg["rebalance"]["cost_bps"]
    w, rb, rows = b.copy(), (build_rebalancer(cfg, px) if name == "riskpulse" else None), []
    for i in range(len(px) - 1):
        now, z_now = px.index[i], None
        seen = sig[sig["ts"] <= now]
        if name == "bench_daily" and (i == 0 or now.date() != px.index[i - 1].date()):
            target = b
        elif name == "naive":
            target = naive_target(b, seen, now, tickers)
        elif name == "riskpulse":
            rb.w = w.copy()                   # real, drifted weights are the starting point
            prev = px.index[i - 1] if i else sig["ts"].min() - pd.Timedelta(days=1)
            new = sig[(sig["ts"] > prev) & (sig["ts"] <= now)]
            out = rb.step(now, seen, px, new)
            target, z_now = (out["weights"] if out else w), rb.z_now
        else:
            target = w                        # bench_hold, or a day with no rebalance
        turn = float(np.abs(target - w).sum())
        r = px.iloc[i + 1] / px.iloc[i] - 1
        gross = float((target * r).sum())
        ic = spearmanr(z_now, r)[0] if z_now is not None else np.nan
        rows.append({"ts": px.index[i + 1], "ret": gross - turn * bps / 1e4,
                     "turn": turn, "cost": turn * bps / 1e4, "ic": ic})
        w = target * (1 + r) / (1 + gross)    # weights drift with returns
    return pd.DataFrame(rows).set_index("ts")

def load_signals():
    sig = pd.read_json("data/signals.jsonl", lines=True)
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True)
    return sig

def load_prices_csv():
    prices = pd.read_csv("data/prices.csv", index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True)
    return prices

def main():
    cfg = load_config()
    try:
        sig = load_signals()
        px = load_prices_csv().dropna(axis=1, how="all").ffill().bfill()
    except Exception as e:
        print("Cannot load data/signals.jsonl or data/prices.csv. Have you run make score?")
        return
        
    start = px.index[int(0.6 * len(px))]
    bpy = len(px) / ((px.index[-1] - px.index[0]).days / 365.25)   # bars per year, from the data
    runs = {k: run_strategy(n, override(cfg, p), sig, px)[lambda d: d.index >= start]
            for k, (n, p) in SCENARIOS.items()}
    bench = runs["Equal-weight, hold"]["ret"]
    table = pd.DataFrame({k: perf(d, bench, bpy) for k, d in runs.items()}).T
    table.round(4).to_csv("data/backtest_results.csv")
    pd.DataFrame({k: (1 + d["ret"]).cumprod() for k, d in runs.items()}).to_csv("data/equity_curves.csv")
    print(table.round(4))

    # Cost sensitivity evaluation across [0, 5, 10, 25] bps
    cost_levels = [0, 5, 10, 25]
    sens_rows = []
    for c_bps in cost_levels:
        c_cfg = copy.deepcopy(cfg)
        c_cfg["rebalance"]["cost_bps"] = c_bps
        res_frust = run_strategy("riskpulse", c_cfg, sig, px)[lambda d: d.index >= start]
        p_frust = perf(res_frust, bench, bpy)
        res_naive = run_strategy("naive", c_cfg, sig, px)[lambda d: d.index >= start]
        p_naive = perf(res_naive, bench, bpy)
        sens_rows.append({
            "cost_bps": c_bps,
            "frustrum_sharpe": round(p_frust["sharpe"], 3),
            "frustrum_ann_ret": round(p_frust["ann_return"], 4),
            "frustrum_cost_drag_bps": round(p_frust["cost_drag_bps_yr"], 1),
            "naive_sharpe": round(p_naive["sharpe"], 3),
            "naive_ann_ret": round(p_naive["ann_return"], 4),
            "naive_cost_drag_bps": round(p_naive["cost_drag_bps_yr"], 1),
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv("data/cost_sensitivity.csv", index=False)
    print("\n--- Cost Sensitivity Table ---")
    print(sens_df)

if __name__ == "__main__":
    main()
