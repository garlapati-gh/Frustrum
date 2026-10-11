import pandas as pd
import json
import plotly.express as px
import requests
import streamlit as st
from src.common import db
from src.common.config import load_config
from src.signals.views import explain

API, cfg = "http://localhost:8000", load_config()
st.set_page_config(page_title="Frustrum", layout="wide")

def get(path, **params):
    try:
        return requests.get(f"{API}{path}", params=params, timeout=5).json()
    except Exception:
        return []

@st.fragment(run_every="2s")
def live():
    h = get("/health")
    w = pd.DataFrame(get("/weights/history"))
    if w.empty:
        st.info("No data yet. Start a replay on the Control tab."); return
    w["ts"] = pd.to_datetime(w["ts"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Simulated time", str(h.get("sim_time", "-"))[:16])
    c2.metric("Rebalances", w["ts"].nunique())
    last = w[w["ts"] == w["ts"].max()].set_index("ticker")
    c3.metric("Largest active weight", f"{(last['weight'] - 1/len(last)).abs().max():.2%}")
    
    # Load sectors to map colors
    sectors = pd.read_csv("data/tickers.csv")[["ticker", "sector"]]
    color_palette = px.colors.qualitative.Pastel
    sec_c = {s: color_palette[i % len(color_palette)] for i, s in enumerate(sectors.sector.unique())}
    color_discrete_map = {r.ticker: sec_c[r.sector] for r in sectors.itertuples()}
    
    fig = px.area(w, x="ts", y="weight", color="ticker", title="Index weights over time", color_discrete_map=color_discrete_map)
    
    # Add vertical lines for circuit breakers if any
    try:
        breakers = db.read("SELECT DISTINCT ts FROM trades WHERE reason LIKE '%breaker%'")
        for b_ts in breakers["ts"]:
            fig.add_vline(x=b_ts, line_dash="dash", line_color="red", annotation_text="Circuit Breaker")
    except Exception:
        pass
        
    st.plotly_chart(fig, use_container_width=True, key="weights_area")
    
    active = (last["weight"] - 1 / len(last)).rename("active weight").reset_index()
    st.plotly_chart(px.bar(active, x="ticker", y="active weight", title="Latest tilt vs equal weight", color="ticker", color_discrete_map=color_discrete_map),
                    use_container_width=True, key="tilt_bar")
    feed = pd.DataFrame(get("/signals", limit=15))
    if not feed.empty:
        st.dataframe(feed[["ts", "ticker", "kind", "event_type", "sentiment", "impact", "text"]],
                     use_container_width=True, hide_index=True)

tab_live, tab_why, tab_perf, tab_ctrl = st.tabs(["Live", "Why", "Performance", "Control"])
with tab_live:
    live()
with tab_why:
    try:
        sig = db.read("SELECT * FROM signals")
        sig["ts"] = pd.to_datetime(sig["ts"], utc=True)
        meta = db.read("SELECT value FROM meta WHERE key='sim_time'")
        if not sig.empty and len(meta):
            t = st.selectbox("Stock", cfg["universe"])
            top = explain(sig, t, pd.Timestamp(meta["value"].iloc[0]), cfg, n=8)
            st.plotly_chart(px.bar(top, x="c", y="text", orientation="h",
                                   title=f"Headlines driving {t}'s view (decayed)"), use_container_width=True)
            st.dataframe(top, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Database empty.")
with tab_perf:
    try:
        res = pd.read_csv("data/backtest_results.csv", index_col=0)
        st.dataframe(res.style.format("{:.3f}"), use_container_width=True)
        st.line_chart(pd.read_csv("data/equity_curves.csv", index_col=0))
        st.bar_chart(res["sharpe"])
    except Exception:
        st.info("Run backtest first to view performance.")
with tab_ctrl:
    try:
        events = json.load(open("data/demo_events.json"))
        ev = st.selectbox("Replay window", events, format_func=lambda e: e["name"])
        start_date = ev.get("start")
        end_date = ev.get("end")
    except Exception:
        events = None
        start_date = None
        end_date = None
        st.warning("No data/demo_events.json found. Will replay all available data.")
        
    speed = st.slider("Replay speed (x real time)", 60, 6000, 1200)
    lam = st.slider("Tilt strength", 0.0, 1.5, cfg["rebalance"]["lambda_tilt"])
    cap = st.slider("Turnover cap per rebalance", 0.01, 0.30, float(cfg["rebalance"]["turnover_cap"]))
    cost = st.slider("Transaction cost (bps)", 0, 25, int(cfg["rebalance"]["cost_bps"]))
    
    if st.button("Start replay"):
        requests.post(f"{API}/replay/start", params={"speed": speed, "start": start_date,
                      "end": end_date, "lambda_tilt": lam, "turnover_cap": cap, "cost_bps": cost})
    if st.button("Stop"):
        requests.post(f"{API}/replay/stop")
