import numpy as np
import pandas as pd

def exp_tilt(b: pd.Series, z: pd.Series, lam: float, max_tilt: float) -> pd.Series:
    w = (b * np.exp(lam * z)).clip(lower=b * (1 - max_tilt), upper=b * (1 + max_tilt))
    return w / w.sum()
