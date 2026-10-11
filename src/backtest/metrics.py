import numpy as np

def perf(df, bench_ret, bpy):
    r = df["ret"]; eq = (1 + r).cumprod()
    ann = eq.iloc[-1] ** (bpy / len(r)) - 1
    vol = r.std() * np.sqrt(bpy)
    te = (r - bench_ret.reindex(r.index)).std() * np.sqrt(bpy)
    return {"ann_return": ann, "ann_vol": vol, "sharpe": ann / vol if vol else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(),
            "annual_turnover": df["turn"].sum() * bpy / len(r),
            "cost_drag_bps_yr": df["cost"].sum() * bpy / len(r) * 1e4,
            "tracking_error": te, "ic_mean": df["ic"].mean()}
