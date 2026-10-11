"""
Full dashboard for Frustrum – event-aware index rebalancer.

Panels:
  1. Live   – weight area chart (sector colours) + news event markers + circuit breakers
  2. Feed   – live signal feed with ticker, sentiment, event type, impact, source
  3. Why    – click a stock, see decayed headline contributions + before/after weights
  4. Perf   – equity curve and backtest metrics table (all strategies + ablations)
  5. NLP    – validation: avg abnormal return after positive vs negative signals
  6. Ctrl   – start/stop replay, tune sliders
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from src.common import db
from src.common.config import load_config
from src.signals.views import explain

API = "http://localhost:8000"
cfg = load_config()
st.set_page_config(page_title="Frustrum – NLP Index Rebalancer", layout="wide")

# ── Sector colour palette ──────────────────────────────────────────────────
_SECTORS_DF = pd.read_csv("data/tickers.csv")[["ticker", "sector"]]
_PALETTE    = px.colors.qualitative.Pastel
_SEC_COLOR  = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(_SECTORS_DF["sector"].unique())}
COLOR_MAP   = {r.ticker: _SEC_COLOR[r.sector] for r in _SECTORS_DF.itertuples()}

def api(path, **params):
    try:
        return requests.get(f"{API}{path}", params=params, timeout=5).json()
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════════════
# TAB DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════
tab_live, tab_feed, tab_why, tab_perf, tab_nlp, tab_ctrl = st.tabs(
    ["📈 Live Weights", "📰 Signal Feed", "❓ Why Changed", "📊 Performance", "🔬 NLP Validation", "⚙️ Control"]
)

# ══════════════════════════════════════════════════════════════════════════
# 1. LIVE WEIGHTS TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_live:
    @st.fragment(run_every="3s")
    def live_weights():
        h = api("/health")
        w = pd.DataFrame(api("/weights/history"))
        if w.empty:
            st.info("No data yet – start a replay in the **Control** tab.")
            return

        w["ts"] = pd.to_datetime(w["ts"])
        c1, c2, c3 = st.columns(3)
        replay_time = h.get("sim_time")
        replay_time = (
            pd.to_datetime(replay_time, utc=True).strftime("%Y-%m-%d %H:%M UTC")
            if replay_time
            else "–"
        )
        c1.metric("Replay time (UTC)", replay_time)
        c2.metric("Rebalances", w["ts"].nunique())
        last = w[w["ts"] == w["ts"].max()].set_index("ticker")
        c3.metric("Largest tilt from equal-weight", f"{(last['weight'] - 1/len(last)).abs().max():.2%}")

        # Stacked area chart with sector colours
        fig = px.area(w, x="ts", y="weight", color="ticker",
                      title="Index weights over time (sector colours)",
                      color_discrete_map=COLOR_MAP,
                      labels={"weight": "Portfolio weight", "ts": ""})

        # Add news-event markers
        try:
            sigs = pd.DataFrame(api("/signals", limit=300))
            if not sigs.empty:
                sigs["ts"] = pd.to_datetime(sigs["ts"])
                # Show one vline per batch of signals within 30-min windows
                sig_times = sigs.groupby(sigs["ts"].dt.floor("30min")).first().reset_index(drop=True)
                for _, row in sig_times.iterrows():
                    color = "orange" if row.get("sentiment", 0) >= 0 else "red"
                    fig.add_vline(x=row["ts"], line_dash="dot", line_color=color,
                                  opacity=0.4, line_width=1)
        except Exception:
            pass

        # Circuit-breaker vlines
        try:
            breakers = db.read("SELECT DISTINCT ts FROM trades WHERE reason='breaker'")
            for b_ts in breakers["ts"]:
                fig.add_vline(x=b_ts, line_dash="dash", line_color="red",
                              annotation_text="⚡ Breaker", line_width=2)
        except Exception:
            pass

        st.plotly_chart(fig, use_container_width=True, key="wgt_area")

        # Current tilt bar
        active = (last["weight"] - 1 / len(last)).rename("active weight").reset_index()
        fig2 = px.bar(active, x="ticker", y="active weight",
                      title="Current tilt vs equal-weight", color="ticker",
                      color_discrete_map=COLOR_MAP)
        fig2.add_hline(y=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig2, use_container_width=True, key="tilt_bar")

    live_weights()

# ══════════════════════════════════════════════════════════════════════════
# 2. SIGNAL FEED TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_feed:
    @st.fragment(run_every="3s")
    def signal_feed():
        sigs = pd.DataFrame(api("/signals", limit=50))
        if sigs.empty:
            st.info("Waiting for signals…")
            return
        sigs["ts"] = pd.to_datetime(sigs["ts"])
        sigs = sigs.sort_values("ts", ascending=False)

        # Colour rows by sentiment
        def row_color(row):
            sentiment = row["sentiment"]
            if sentiment > 0.2:
                color = "background-color: #d4edda"
            elif sentiment < -0.2:
                color = "background-color: #f8d7da"
            else:
                color = ""
            return [color] * len(row)

        cols_show = ["ts", "ticker", "kind", "source", "event_type", "sentiment", "impact", "confidence", "text"]
        cols_show = [c for c in cols_show if c in sigs.columns]
        styled = sigs[cols_show].style.apply(row_color, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    signal_feed()

# ══════════════════════════════════════════════════════════════════════════
# 3. WHY CHANGED TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_why:
    ticker_sel = st.selectbox("Select stock", cfg["universe"], key="why_ticker")
    try:
        all_sigs = db.read("SELECT * FROM signals")
        if all_sigs.empty:
            st.info("Run a replay first.")
        else:
            all_sigs["ts"] = pd.to_datetime(all_sigs["ts"], utc=True)
            meta = db.read("SELECT value FROM meta WHERE key='sim_time'")
            now_str = meta["value"].iloc[0] if len(meta) else None
            if now_str:
                now_ts = pd.Timestamp(now_str, tz="UTC")
                top = explain(all_sigs, ticker_sel, now_ts, cfg, n=10)
                if not top.empty:
                    fig = px.bar(top, x="c", y="text", orientation="h",
                                 title=f"Top decayed contributions to {ticker_sel}'s view",
                                 labels={"c": "Decayed contribution", "text": ""})
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(top[["ts", "event_type", "sentiment", "impact", "novelty", "c", "text"]],
                                 use_container_width=True, hide_index=True)

                # Show before/after weights
                wdf = db.read("""
                    SELECT ts, weight FROM weights
                    WHERE ticker=? ORDER BY ts
                """, (ticker_sel,))
                if not wdf.empty:
                    wdf["ts"] = pd.to_datetime(wdf["ts"])
                    wdf["benchmark"] = 1.0 / len(cfg["universe"])
                    fig2 = px.line(wdf, x="ts", y=["weight", "benchmark"],
                                   title=f"{ticker_sel} weight over time vs equal-weight benchmark")
                    st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.info(f"Database empty or replay not started. ({e})")

# ══════════════════════════════════════════════════════════════════════════
# 4. PERFORMANCE TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_perf:
    try:
        res = pd.read_csv("data/backtest_results.csv", index_col=0)
        st.subheader("Strategy comparison (2-year backtest on synthetic data)")
        # Highlight Frustrum rows
        def highlight_frustrum(row):
            if "Frustrum" in str(row.name) and "no" not in str(row.name).lower():
                return ["background-color: #cce5ff; font-weight: bold"] * len(row)
            return [""] * len(row)
        st.dataframe(res.style.apply(highlight_frustrum, axis=1).format("{:.4f}"),
                     use_container_width=True)

        # Equity curves
        eq = pd.read_csv("data/equity_curves.csv", index_col=0)
        eq.index = pd.to_datetime(eq.index)
        fig = px.line(eq, title="Equity curves – growth of $1 invested",
                      labels={"value": "Portfolio value ($)", "index": ""})
        st.plotly_chart(fig, use_container_width=True)

        # Ablation: Sharpe
        ablation_cols = [c for c in res.index if "Frustrum" in c]
        if ablation_cols:
            abl_df = res.loc[ablation_cols, ["ann_return", "sharpe", "max_drawdown", "annual_turnover"]]
            fig2 = px.bar(abl_df.reset_index(), x="index", y="sharpe",
                          title="Sharpe ratio – ablation study",
                          labels={"index": "Strategy", "sharpe": "Sharpe ratio"})
            st.plotly_chart(fig2, use_container_width=True)
    except FileNotFoundError:
        st.info("Run `make backtest` to generate performance results.")
    except Exception as e:
        st.warning(f"Error loading results: {e}")

# ══════════════════════════════════════════════════════════════════════════
# 5. NLP VALIDATION TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_nlp:
    st.subheader("NLP Signal Validation – Average Abnormal Return by Sentiment Quintile")
    st.markdown("""
    We check whether the NLP-scored sentiment predicts next-day stock returns.
    Signals are binned into quintiles by sentiment score. If the model has skill,
    the strongly-negative quintile (Q1) should have lower returns than Q5 (strongly positive).
    """)
    try:
        sigs_df = pd.read_json("data/signals.jsonl", lines=True)
        px_df = pd.read_csv("data/prices.csv", index_col=0)
        px_df.index = pd.to_datetime(px_df.index, utc=True)
        px_df = px_df.ffill().bfill()

        sigs_df["ts"] = pd.to_datetime(sigs_df["ts"], utc=True)
        sigs_df = sigs_df[sigs_df["ticker"].isin(px_df.columns)]

        # Compute next-day return for each signal
        rets = px_df.pct_change().shift(-1)
        eq_ret = rets.mean(axis=1)  # equal-weight market return
        abnormal = rets.subtract(eq_ret, axis=0)  # stock minus market

        def get_fwd_return(row):
            t = row["ticker"]
            if t not in abnormal.columns:
                return float("nan")
            # Find next price date
            mask = abnormal.index.date > row["ts"].date()
            future = abnormal.index[mask]
            if len(future) == 0:
                return float("nan")
            return abnormal.loc[future[0], t]

        if len(sigs_df) > 0:
            sample = sigs_df[sigs_df["ticker"] != "*"].copy()
            if len(sample) > 300:
                sample = sample.sample(300, random_state=42)
            sample["fwd_abnormal_ret"] = sample.apply(get_fwd_return, axis=1)
            sample = sample.dropna(subset=["fwd_abnormal_ret"])

            if len(sample) >= 5:
                sample["quintile"] = pd.qcut(sample["sentiment"], 5, labels=["Q1 (most neg)", "Q2", "Q3", "Q4", "Q5 (most pos)"])
                agg = sample.groupby("quintile")["fwd_abnormal_ret"].mean().reset_index()
                agg.columns = ["Sentiment quintile", "Avg next-day abnormal return"]
                fig = px.bar(agg, x="Sentiment quintile", y="Avg next-day abnormal return",
                             title="Avg abnormal return (t+1) by signal sentiment quintile",
                             color="Avg next-day abnormal return",
                             color_continuous_scale=["red", "lightgray", "green"])
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Upward slope (Q1 < Q5) indicates NLP sentiment has directional predictive value.")
                st.dataframe(agg, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough valid signal-return pairs to compute quintiles.")
    except Exception as e:
        st.info(f"Run `make score` and `make backtest` first. ({e})")

# ══════════════════════════════════════════════════════════════════════════
# 6. CONTROL TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_ctrl:
    st.subheader("Replay Control")
    st.caption(
        "Replay timestamps come from the historical dataset (UTC), not your computer's live clock. "
        "Starting a replay clears the previous run."
    )

    col1, col2 = st.columns(2)
    with col1:
        try:
            events = json.load(open("data/demo_events.json"))
            ev = st.selectbox("Demo window", events, format_func=lambda e: e["name"])
            start_date, end_date = ev.get("start"), ev.get("end")
        except Exception:
            st.warning("No data/demo_events.json found. Replaying all data.")
            start_date, end_date = None, None

        speed = st.slider("Replay speed (x real time)", 60, 3600, 600)
        lam   = st.slider("Tilt strength (λ)", 0.0, 1.5, float(cfg["rebalance"]["lambda_tilt"]))
        cap   = st.slider("Turnover cap per rebalance", 0.01, 0.30, float(cfg["rebalance"]["turnover_cap"]))
        cost  = st.slider("Transaction cost (bps)", 0, 25, int(cfg["rebalance"]["cost_bps"]))

    with col2:
        st.markdown("**Legend**")
        st.markdown("🟠 Dotted orange lines = positive news event")
        st.markdown("🔴 Dotted red lines = negative news event")
        st.markdown("⚡ Solid red lines = circuit breaker fired")
        st.markdown("---")
        st.markdown("**Engine config (live)**")
        st.json({
            "cost_bps": cost, "turnover_cap": cap,
            "lambda_tilt": lam, "view_trigger": cfg["rebalance"]["view_trigger"],
            "cooldown_min": cfg["rebalance"]["cooldown_min"],
        })

    c1, c2 = st.columns(2)
    if c1.button("▶ Start replay", type="primary"):
        try:
            response = requests.post(
                f"{API}/replay/start",
                params=dict(
                    speed=speed,
                    start=start_date,
                    end=end_date,
                    lambda_tilt=lam,
                    turnover_cap=cap,
                    cost_bps=cost,
                ),
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"Could not start replay. Check that the backend is running at {API}. ({exc})")
        else:
            st.success("Replay started! Switch to the Live Weights tab.")
    if c2.button("⏹ Stop"):
        requests.post(f"{API}/replay/stop")
        st.info("Replay stopped.")

    # Engine output example
    st.subheader("Engine output example")
    try:
        sigs_ex = db.read("SELECT * FROM signals ORDER BY ts DESC LIMIT 5")
        if not sigs_ex.empty:
            st.json(sigs_ex.to_dict("records"))
    except Exception:
        st.caption("(Start a replay to see live engine output here.)")
