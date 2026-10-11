from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import pytest
from src.signals.views import contributions, build_views
from src.common.config import load_config

@pytest.fixture
def base_cfg():
    cfg = load_config()
    cfg["ablation"]["decay"] = True
    cfg["ablation"]["dedup"] = True
    cfg["ablation"]["contagion"] = True
    return cfg

def test_half_life_decay(base_cfg):
    now = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    # earnings half life is 1 day
    hl_days = base_cfg["half_life_days"]["earnings"]
    ts_half = now - timedelta(days=hl_days)
    ts_now = now

    df = pd.DataFrame([
        {"ts": ts_now, "event_type": "earnings", "sentiment": 1.0, "impact": 10, "confidence": 1.0, "novelty": 1.0, "kind": "news"},
        {"ts": ts_half, "event_type": "earnings", "sentiment": 1.0, "impact": 10, "confidence": 1.0, "novelty": 1.0, "kind": "news"}
    ])
    
    contribs = contributions(df, now, base_cfg)
    assert np.isclose(contribs.iloc[0], 1.0)
    assert np.isclose(contribs.iloc[1], 0.5)

def test_zero_novelty_contributes_zero(base_cfg):
    now = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        {"ts": now, "event_type": "product", "sentiment": 0.8, "impact": 8, "confidence": 0.9, "novelty": 0.0, "kind": "news"}
    ])
    contribs = contributions(df, now, base_cfg)
    assert contribs.iloc[0] == 0.0

def test_no_contagion_equals_direct(base_cfg):
    now = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    tickers = ["AAPL", "MSFT"]
    sector_of = {"AAPL": "Technology", "MSFT": "Technology"}
    sector_beta = {}
    corr = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=tickers, columns=tickers)

    signals = pd.DataFrame([
        {"ts": now, "ticker": "AAPL", "event_type": "earnings", "sentiment": 1.0, "impact": 10, "confidence": 0.9, "novelty": 1.0, "kind": "news"}
    ])

    base_cfg["ablation"]["contagion"] = False
    views = build_views(signals, now, base_cfg, tickers, sector_of, sector_beta, corr)
    assert views["AAPL"] > 0
    assert views["MSFT"] == 0.0

def test_geopolitical_macro_spillover(base_cfg):
    now = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    tickers = ["XOM", "AAPL"]
    sector_of = {"XOM": "Energy", "AAPL": "Technology"}
    # Negative geopolitical event for energy sector might have positive beta (e.g. oil price spike)
    sector_beta = {"Energy": {"geopolitical": -1.5}, "Technology": {"geopolitical": 0.2}}
    corr = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=tickers, columns=tickers)

    # Negative geopolitical story at ticker '*'
    signals = pd.DataFrame([
        {"ts": now, "ticker": "*", "event_type": "geopolitical", "sentiment": -1.0, "impact": 10, "confidence": 1.0, "novelty": 1.0, "kind": "news"}
    ])

    base_cfg["ablation"]["contagion"] = True
    views = build_views(signals, now, base_cfg, tickers, sector_of, sector_beta, corr)
    # Energy has negative beta to negative sentiment -> positive view impact
    assert views["XOM"] > 0
    assert views["AAPL"] < 0

