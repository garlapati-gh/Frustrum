from datetime import timedelta
import numpy as np

class Guard:
    def __init__(self, tickers, cfg):
        self.t, self.cfg = list(tickers), cfg
        self.last_change = {t: None for t in self.t}
        self.breaker_until = {t: None for t in self.t}

    def update_breaker(self, new_signals, now) -> bool:
        cb, fired = self.cfg["circuit_breaker"], False
        for r in new_signals.itertuples():
            if (r.ticker in self.breaker_until and r.impact >= cb["impact_min"]
                    and r.sentiment <= cb["sentiment_max"] and r.confidence >= 0.5):
                self.breaker_until[r.ticker] = now + timedelta(minutes=cb["freeze_min"])
                fired = True
        return fired

    def bounds(self, now, b, w_prev):
        rb = self.cfg["rebalance"]
        lo = np.maximum(rb["weight_min"], b * (1 - rb["max_tilt"]))
        hi = np.minimum(rb["weight_max"], b * (1 + rb["max_tilt"]))
        for i, t in enumerate(self.t):
            until, last = self.breaker_until[t], self.last_change[t]
            if until and now < until:                       # breaker: pin at the floor
                lo[i] = hi[i] = rb["weight_min"]
            elif last and now - last < timedelta(minutes=rb["cooldown_min"]):
                lo[i] = hi[i] = w_prev[i]                   # cooldown: hold
        return lo, hi
