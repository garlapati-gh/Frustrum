import numpy as np
import pandas as pd

def contributions(s: pd.DataFrame, now, cfg) -> pd.Series:
    age = (now - s["ts"]).dt.total_seconds() / 86400
    hl = s["event_type"].map(cfg["half_life_days"]).fillna(cfg["half_life_days"]["other"])
    dec = 0.5 ** (age / hl) if cfg["ablation"]["decay"] else 1.0
    nov = s["novelty"] if cfg["ablation"]["dedup"] else 1.0
    src = s["kind"].map(cfg["source_weight"])
    return s["sentiment"] * (s["impact"] / 10) * s["confidence"] * nov * src * dec

def build_views(signals, now, cfg, tickers, sector_of, sector_beta, corr) -> pd.Series:
    s = signals[(signals["ts"] <= now) &
                (signals["confidence"] >= cfg["engine"]["min_confidence"])].copy()
    if s.empty:
        return pd.Series(0.0, index=tickers)
    s["c"] = contributions(s, now, cfg)
    stock = s[s["ticker"] != "*"]
    news = stock[stock["kind"] == "news"].groupby("ticker")["c"].sum().reindex(tickers, fill_value=0.0)
    soc = stock[stock["kind"] == "social"].groupby("ticker")["c"].sum().reindex(tickers, fill_value=0.0)
    guard = np.where((np.sign(news) != np.sign(soc)) & (soc.abs() > news.abs()) & (news != 0), 0.5, 1.0)
    direct = news + guard * soc
    if not cfg["ablation"]["contagion"]:
        return direct

    rho = corr.reindex(index=tickers, columns=tickers).clip(lower=0).fillna(0).values.copy()
    np.fill_diagonal(rho, 0.0)
    row_sums = rho.sum(axis=1, keepdims=True)
    w = rho / np.where(row_sums == 0, 1e-9, row_sums)
    peer = pd.Series(w @ direct.values, index=tickers)

    broad = s[(s["ticker"] == "*") & s["event_type"].isin(["macro", "geopolitical"])]
    macro_part = pd.Series(0.0, index=tickers)
    for e in ("macro", "geopolitical"):
        total = broad.loc[broad["event_type"] == e, "c"].sum() if not broad.empty else 0.0
        macro_part += total * pd.Series({t: sector_beta.get(sector_of.get(t), {}).get(e, 0.0) for t in tickers})

    sp = cfg["spillover"]
    return direct + sp["peer_lambda"] * peer + sp.get("broad_scale", 0.5) * macro_part

def to_z(view: pd.Series, cfg) -> pd.Series:
    return (view / cfg.get("view_scale", 0.4)).clip(-3, 3)

def explain(signals, ticker, now, cfg, n=5) -> pd.DataFrame:
    s = signals[(signals["ticker"] == ticker) & (signals["ts"] <= now)].copy()
    if s.empty: return pd.DataFrame()
    s["c"] = contributions(s, now, cfg)
    return s.reindex(s["c"].abs().sort_values(ascending=False).index).head(n)[
        ["ts", "text", "event_type", "sentiment", "impact", "c"]]
