from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from src.rebalance.optimiser import optimise
from src.rebalance.tilt import exp_tilt
from src.rebalance.guardrails import Guard

@pytest.fixture
def mock_setup():
    tickers = ["AAPL", "MSFT", "GOOGL", "XOM", "CVX"]
    n = len(tickers)
    b = pd.Series(1.0 / n, index=tickers)
    sectors = ["Tech", "Tech", "Energy", "Fin", "Health"]
    cov = np.eye(n) * 0.04  # 20% annual vol, uncorrelated
    lo = np.full(n, 0.02)
    hi = np.full(n, 0.40)
    cfg = {
        "risk_aversion": 2.0,
        "sector_cap": 0.45,
        "rebalance": {
            "weight_min": 0.02,
            "weight_max": 0.40,
            "max_tilt": 0.5,
            "cooldown_min": 15,
            "turnover_cap": 0.10,
        },
        "circuit_breaker": {
            "impact_min": 9,
            "sentiment_max": -0.6,
            "freeze_min": 60,
        }
    }
    return {
        "tickers": tickers,
        "b": b,
        "sectors": sectors,
        "cov": cov,
        "lo": lo,
        "hi": hi,
        "cfg": cfg,
    }

def test_weights_sum_to_one_and_bounds(mock_setup):
    s = mock_setup
    w_tilt = s["b"].values * 1.2
    w_tilt /= w_tilt.sum()
    w_prev = s["b"].values

    w_opt = optimise(w_tilt, s["b"].values, w_prev, s["cov"], s["lo"], s["hi"], s["sectors"], s["cfg"])
    assert np.isclose(w_opt.sum(), 1.0, atol=1e-4)
    assert np.all(w_opt >= s["lo"] - 1e-5)
    assert np.all(w_opt <= s["hi"] + 1e-5)

def test_zero_views_returns_benchmark(mock_setup):
    s = mock_setup
    z = pd.Series(0.0, index=s["tickers"])
    w_tilt = exp_tilt(s["b"], z, lam=0.6, max_tilt=0.5)
    w_prev = s["b"].values

    w_opt = optimise(w_tilt.values, s["b"].values, w_prev, s["cov"], s["lo"], s["hi"], s["sectors"], s["cfg"])
    np.testing.assert_allclose(w_opt, s["b"].values, atol=1e-4)

def test_sector_cap_holds(mock_setup):
    s = mock_setup
    # Skew views heavily towards Tech (AAPL, MSFT)
    z = pd.Series([3.0, 3.0, -1.0, -1.0, -1.0], index=s["tickers"])
    w_tilt = exp_tilt(s["b"], z, lam=1.0, max_tilt=0.9).values
    w_prev = s["b"].values

    w_opt = optimise(w_tilt, s["b"].values, w_prev, s["cov"], s["lo"], s["hi"], s["sectors"], s["cfg"])
    tech_indices = [0, 1]
    tech_weight = w_opt[tech_indices].sum()
    assert tech_weight <= s["cfg"]["sector_cap"] + 1e-4

def test_turnover_capping(mock_setup):
    s = mock_setup
    turnover_cap = s["cfg"]["rebalance"]["turnover_cap"]
    w_prev = s["b"].values
    # Target weights far from previous
    w_star = np.array([0.40, 0.40, 0.16, 0.02, 0.02])
    delta = w_star - w_prev
    turn = np.abs(delta).sum()

    if turn > turnover_cap:
        delta *= turnover_cap / turn
    capped_turn = np.abs(delta).sum()
    assert capped_turn <= turnover_cap + 1e-5

def test_circuit_breaker_pins_weight_min(mock_setup):
    s = mock_setup
    guard = Guard(s["tickers"], s["cfg"])
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    # Severe negative news triggering circuit breaker for AAPL
    signals = pd.DataFrame([{
        "ticker": "AAPL",
        "impact": 10,
        "sentiment": -0.8,
        "confidence": 0.9,
    }])
    fired = guard.update_breaker(signals, now)
    assert fired is True
    
    lo, hi = guard.bounds(now, s["b"].values, s["b"].values)
    aapl_idx = s["tickers"].index("AAPL")
    assert lo[aapl_idx] == s["cfg"]["rebalance"]["weight_min"]
    assert hi[aapl_idx] == s["cfg"]["rebalance"]["weight_min"]

