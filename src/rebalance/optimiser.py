import numpy as np
import pandas as pd
from scipy.optimize import minimize

def daily_cov(prices: pd.DataFrame, now, window: int = 60) -> np.ndarray:
    daily = prices[prices.index <= now].resample("1D").last().dropna(how="all")
    return daily.pct_change().dropna().tail(window).cov().values * 252

def optimise(w_tilt, b, w_prev, cov, lo, hi, sectors, cfg) -> np.ndarray:
    gamma, n = cfg["risk_aversion"], len(b)
    f = lambda w: np.sum((w - w_tilt) ** 2) + gamma * (w - b) @ cov @ (w - b)
    g = lambda w: 2 * (w - w_tilt) + 2 * gamma * cov @ (w - b)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    for sec in set(sectors):
        idx = np.where(np.array(sectors) == sec)[0]
        cons.append({"type": "ineq", "fun": lambda w, idx=idx: cfg["sector_cap"] - w[idx].sum()})
    res = minimize(f, w_prev, jac=g, bounds=list(zip(lo, hi)), constraints=cons, method="SLSQP")
    return res.x if res.success else w_prev
