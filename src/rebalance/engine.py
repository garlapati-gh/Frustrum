from datetime import timedelta
import numpy as np
import pandas as pd
from src.signals.views import build_views, to_z
from src.rebalance.tilt import exp_tilt
from src.rebalance.optimiser import daily_cov, optimise
from src.rebalance.guardrails import Guard

class Rebalancer:
    def __init__(self, cfg, tickers, sector_of, sector_beta, bench: pd.Series):
        self.cfg, self.t, self.bench = cfg, list(tickers), bench
        self.sector_of, self.sector_beta = sector_of, sector_beta
        self.w, self.z_last, self.last_run = bench.copy(), pd.Series(0.0, index=self.t), None
        self.z_now = pd.Series(0.0, index=self.t)
        self.guard, self.cum_cost_bps = Guard(self.t, cfg), 0.0

    def step(self, now, signals, prices, new_signals):
        rb = self.cfg["rebalance"]
        px_slice = prices[prices.index <= now]
        if len(px_slice) < 20:
            return None
        fired = self.guard.update_breaker(new_signals, now)
        daily_px = px_slice.resample("1D").last().dropna(how="all")
        if len(daily_px) < 10:
            return None
        corr = daily_px.pct_change().dropna().tail(rb.get("corr_days", 60)).corr()
        if corr.isna().all().all() or corr.empty:
            corr = pd.DataFrame(np.eye(len(self.t)), index=self.t, columns=self.t)
        else:
            corr = corr.fillna(0.0)
        z = to_z(build_views(signals, now, self.cfg, self.t, self.sector_of,
                             self.sector_beta, corr), self.cfg)
        self.z_now = z
        moved = (z - self.z_last).abs().max() >= rb["view_trigger"]
        stale = self.last_run is None or now - self.last_run >= timedelta(minutes=rb["max_gap_min"])
        if not (fired or moved or stale):
            return None
        self.z_last, self.last_run = z, now

        w_tilt = exp_tilt(self.bench, z, rb["lambda_tilt"], rb["max_tilt"])
        lo, hi = self.guard.bounds(now, self.bench.values, self.w.values)
        if lo.sum() > 1 or hi.sum() < 1:
            return None                                     # infeasible: hold
        if rb.get("stage", "B") == "A":
            w_star = np.clip(w_tilt.values, lo, hi); w_star /= w_star.sum()
        else:
            w_star = optimise(w_tilt.values, self.bench.values, self.w.values,
                              daily_cov(prices, now), lo, hi,
                              [self.sector_of[t] for t in self.t], rb)
        delta = w_star - self.w.values
        turn = np.abs(delta).sum()
        cost_ctrl = self.cfg.get("ablation", {}).get("cost_control", True)
        if cost_ctrl and not fired and turn > rb["turnover_cap"]:
            delta *= rb["turnover_cap"] / turn               # scale towards the old weights
            turn = np.abs(delta).sum()
        gain_bps = float(delta @ z.values) * rb["alpha_bps_per_z"]
        cost_bps = turn * rb["cost_bps"]
        if cost_ctrl and not fired and (turn < rb["no_trade_band"] or gain_bps <= cost_bps):
            return None                                     # not worth trading
        w_prev = self.w.copy()
        self.w = pd.Series(self.w.values + delta, index=self.t)
        for i, t in enumerate(self.t):
            if abs(delta[i]) > 1e-4:
                self.guard.last_change[t] = now
        self.cum_cost_bps += cost_bps
        return {"ts": now, "weights": self.w.copy(), "w_prev": w_prev, "delta": pd.Series(delta, index=self.t), "z": z, "turnover": turn,
                "cost_bps": cost_bps,
                "reason": "breaker" if fired else ("view_move" if moved else "scheduled")}
