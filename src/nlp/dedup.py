from datetime import timedelta

class NoveltyIndex:
    def __init__(self, window_hours: int = 24, thr: float = 0.88):
        self.window = timedelta(hours=window_hours)
        self.thr = thr
        self.items = []

    def check(self, ts, ticker, vec) -> float:
        self.items = [x for x in self.items if x[0] >= ts - self.window]
        top = max((float(vec @ v) for _, t, v in self.items if t == ticker), default=0.0)
        self.items.append((ts, ticker, vec))
        return 0.0 if top >= self.thr else 1.0 - top

novelty = NoveltyIndex()
