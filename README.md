# Frustrum
## Module A: AI/NLP Risk Engine & Event-Aware Tactical Index Rebalancer

**Candidate Name:** Gagana Harshini Garlapati
**College Email ID:** lit2024017@iiitl.ac.in
**College / Campus:** IIIT Lucknow
**Module:** Module A: AI/NLP Risk Engine & Tactical Stock Index Rebalancer
**Demo Video (YouTube Unlisted):** TODO
**Presentation Deck:**   https://drive.google.com/file/d/1qDFOqGGCMK4S6gIWz3vnLxYys0wH82B6/view?usp=sharing
**Source Repository:** https://github.com/garlapati-gh/IIITL-Gagana-Hackathon

---

## 1. Project Overview / Problem Statement & Approach

News moves markets, but news is noisy, people overreact, and every trade costs money. Frustrum is an event-aware, cost-aware rebalancer for a **15-stock mock index**. It treats NLP output on financial news as noisy *views* about stocks and turns those views into small, bounded, costed weight changes.

The core idea: **a headline does not set a weight, it sets a belief.** Each story is scored for sentiment, event type, impact (1-10), confidence and novelty. Scores are faded over time at an event-specific half-life, combined into one view per stock, and converted into target weights that stay close to an equal-weight benchmark. A trade only happens if guardrails allow it and the expected gain beats the transaction cost.

The system has three parts:

| Part | Job | Where |
|---|---|---|
| The reader | Scores each story: sentiment, event type, impact, confidence, novelty | `src/nlp/` |
| The decision maker | Builds views, target weights and applies guardrails | `src/signals/`, `src/rebalance/` |
| The scoreboard | Dashboard with explanations, plus a replay backtest vs baselines | `src/dashboard/`, `src/backtest/` |

**Key ideas**

1. **Signal fusion.** View = sentiment x source weight x (impact / 10) x confidence x novelty x time decay.
2. **Event-type half-lives.** Earnings fade in about 1 day; credit events in about 10 days.
3. **Novelty filter.** Near-duplicate stories (embedding similarity >= 0.88 within 24 h) get novelty 0, so reposts count once.
4. **Social-vs-news guard.** If social disagrees with news and is larger, its push is halved.
5. **Contagion.** Part of each stock's view spills to correlated peers; macro and geopolitical stories spread to sectors through a sensitivity table.
6. **Optimiser with guardrails.** Exponential tilt of the benchmark, then an SLSQP optimiser that penalises active risk, with weight bounds, sector cap, turnover cap, cooldown and a circuit breaker.
7. **Explain and prove.** A "Why Changed" tab links weight moves to headlines; a backtest compares against baselines with ablations and a cost-sensitivity table.

This is a simulation. There is no real money, no exchange connection and no promise of profit. It is not financial advice.

## 2. Architecture & Tech Stack


**Data flow:** news text -> NLP engine (ticker linking, FinBERT sentiment, event classification, novelty, impact, confidence) -> signals (SQLite and `data/signals.jsonl`) served by FastAPI -> rebalancer (views, tilt, optimiser, guardrails, trade decision) -> decision log -> Streamlit dashboard and backtest.

| Layer | Choice |
|---|---|
| Language | Python |
| Sentiment | FinBERT (`ProsusAI/finbert`), CPU |
| Event classification and novelty | `sentence-transformers/all-MiniLM-L6-v2`, prototype similarity plus keyword boosts |
| Impact | Transparent rule-based score (see below) |
| Optimiser | SciPy SLSQP |
| Storage | SQLite (`data/frustrum.db`) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Prices | yfinance |
| Config | `config.yaml` |

### Signal record

| Field | Range | How it is produced |
|---|---|---|
| `sentiment` | -1 to 1 | FinBERT: P(positive) minus P(negative) |
| `event_type` | 9 classes | Embedding similarity to prototypes in `data/event_prototypes.json`, plus keyword boosts |
| `impact` | 1 to 10 | Rule-based (below) |
| `confidence` | 0 to 1 | 0.4 x FinBERT confidence + 0.3 x event match + 0.3 x source credibility |
| `novelty` | 0 to 1 | 1 minus highest similarity to same-ticker stories in the last 24 h; 0 if >= 0.88 |
| `ticker`, `kind`, `source`, `ts`, `text`, `url` | n/a | Used by the explain panel |

**Impact score (rule-based, not trained):**
`impact = BASE[event_type] + 2 x |sentiment| (+2 for fraud / bankruptcy / halt, else +1 for other severe cues such as billion, recall, probe, ban, guidance) - 1 if social`, rounded and clipped to 1-10. Base scores: credit / M&A / geopolitical 7; regulatory / macro / earnings 6; management 5; product 4; other 1.

### Event half-lives (days)

| Earnings | Product | Management | M&A | Regulatory | Credit | Macro | Geopolitical | Other |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 5 | 5 | 10 | 3 | 7 | 0.5 |

### From views to weights

1. **View per stock:** decayed contributions are summed (news and social separately), plus peer spillover (rolling 60-day positive correlation, factor 0.3) and sector spillover for macro and geopolitical stories (factor 0.5).
2. **Scale:** `z = clip(view / 0.4, -3, 3)`.
3. **Target:** `w_tilde_i = b_i * exp(0.6 * z_i) / sum_j(b_j * exp(0.6 * z_j))`, kept within +/-50% of the benchmark weight.
4. **Stage B (default):** SLSQP finds weights close to the target that minimise active risk `(w - b)' Sigma (w - b)` (risk aversion 2.0) subject to bounds, 100% total and a 35% sector cap. Stage A (plain clipped tilt) is available via `rebalance.stage: A`. With no news, weights stay at the benchmark.
5. **Trade decision:** the rebalancer wakes when the circuit breaker fires, when any z-score moves by at least **0.20**, or after 60 minutes. Turnover is capped, moves below **0.5%** are skipped, and the trade needs expected gain (`delta . z x 10 bps`) to exceed cost.

### Guardrails (all in `config.yaml`)

| Guardrail | Default |
|---|---|
| Weight band per stock | 2% to 12% |
| Max tilt vs benchmark | +/-50% of benchmark weight |
| Sector cap | 35% |
| Turnover cap per rebalance | 10% |
| Transaction cost | 5 bps |
| No-trade band | 0.5% total move |
| Cooldown per stock | 15 minutes |
| Circuit breaker | Impact >= 9, sentiment <= -0.6, confidence >= 0.5: pin stock at 2% floor for 60 minutes; bypasses turnover cap and trade-cost test |

### Dashboard (Streamlit, six tabs)

1. **Live Weights:** weights over time with news-event and circuit-breaker markers.
2. **Signal Feed:** latest scored signals.
3. **Why Changed:** headlines behind a stock's latest weight change.
4. **Performance:** backtest comparison table and charts.
5. **NLP Validation:** average abnormal return by sentiment quintile.
6. **Control:** pick replay window and speed, adjust tilt strength, turnover cap and cost, start / stop.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Status, simulated time, last error |
| POST | `/score` | Score one headline interactively |
| GET | `/signals` | Signals (`ticker`, `since`, `limit`) |
| GET | `/weights/latest` | Latest weights and z-scores |
| GET | `/weights/history` | Weight history |
| GET | `/trades` | Per-ticker trade log with headline |
| GET | `/explain/{ticker}` | Latest records for a stock |
| POST | `/replay/start` | Start replay (`speed`, `start`, `end`, `lambda_tilt`, `turnover_cap`, `cost_bps`) |
| POST | `/replay/stop` | Stop replay |

Interactive docs: http://localhost:8000/docs

### Repository structure

```text
├── README.md  requirements.txt  LICENSE  Makefile  config.yaml
├── data/       tickers.csv, sector_sensitivity.csv, event_prototypes.json,
│               demo_events.json, sample_news.jsonl, prices_sample.csv
├── scripts/    download_data.py, precompute_scores.py, train_impact.py
├── src/
│   ├── common/     config.py, schemas.py, db.py
│   ├── ingest/     gdelt.py, prices.py, replay.py
│   ├── nlp/        entity.py, dedup.py, sentiment.py, events.py, impact.py, pipeline.py
│   ├── signals/    views.py
│   ├── rebalance/  tilt.py, optimiser.py, guardrails.py, engine.py
│   ├── api/        main.py
│   ├── backtest/   run.py, metrics.py
│   ├── dashboard/  app.py
│   └── runner.py
├── tests/      test_decay.py, test_optimiser.py, test_pipeline.py
└── docs/       presentation.pdf, architecture.png
```

## 3. Dataset Used

| Data | Source | Notes |
|---|---|---|
| Prices | yfinance, 2 years of **daily** bars for the 15 tickers | Saved to `data/prices.csv`; last 30 days committed as `data/prices_sample.csv` |
| News and social text | **Synthetic**, generated by `scripts/download_data.py` (seed 42) from labelled headline templates per event type, across the 2-year price range | Clearly synthetic. Includes a dense 14-day demo window and one planted circuit-breaker event (a severe fraud-probe headline for BA). First 200 rows committed as `data/sample_news.jsonl` |
| Reference tables | Hand-written | `tickers.csv` (names, aliases, sector, peers), `sector_sensitivity.csv`, `event_prototypes.json` |

The repo also contains a GDELT DOC 2.0 fetcher (`src/ingest/gdelt.py`), but the committed pipeline does **not** use it: all scored news is the synthetic set above. No real tweet, Kaggle, NewsAPI or Reddit data is used. No confidential or proprietary S&P Global or CRISIL data is used.

Because the news is synthetic and its dates are drawn at random, backtest and validation results demonstrate that the **pipeline works end to end**, not that news predicts returns in real markets.

**Universe:** AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, BAC, XOM, CVX, JNJ, PFE, WMT, KO, BA, DIS. Benchmark is equal weight (6.67% each).

### Assumptions (hand-set starting values)

- Equal-weight benchmark.
- Half-life per event type (table above).
- Sector sensitivity (macro / geopolitical exposure to good news): Technology 0.8 / 0.5; Communication 0.6 / 0.4; Consumer Discretionary 0.8 / 0.5; Financials 1.0 / 0.5; Energy 0.3 / -0.6; Industrials 0.7 / 0.2; Health Care 0.3 / 0.2; Consumer Staples 0.2 / 0.2.
- Expected return of 10 bps per unit of z-score (`alpha_bps_per_z`).
- Transaction cost of 5 bps.
- Source credibility (confidence): GDELT / NewsAPI / synthetic news 0.8, tweets / synthetic social 0.5, Reddit 0.4. Source weight in views: news 1.0, social 0.5.
- Confidence weights 0.4 / 0.3 / 0.3; `view_scale` 0.4; dedup threshold 0.88; tilt strength 0.6; risk aversion 2.0.
- Impact base scores and keyword cues.
- All guardrail values.
- Sharpe uses a zero risk-free rate.

## 4. Quickstart & Installation

**Prerequisites:** Python 3.10-3.12, Git, a browser. Around 8 GB RAM and 3-5 GB disk recommended. CPU only.
**Tested on:** TODO (state your OS).
**Internet:** needed for `pip`, the one-time FinBERT and MiniLM downloads, and the yfinance price download. Afterwards, replay and the dashboard run from saved files.

```bash
git clone https://github.com/garlapati-gh/IIITL-Gagana-Hackathon.git
cd IIITL-Gagana-Hackathon

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

make setup                          # pip install -r requirements.txt
make data                           # prices (yfinance) + synthetic news -> data/raw, data/prices.csv
make score                          # NLP engine -> data/signals.jsonl
make backtest                       # -> data/backtest_results.csv, equity_curves.csv, cost_sensitivity.csv
```

Run the demo in two terminals:

```bash
make run-api                        # FastAPI on http://localhost:8000
make dashboard                      # Streamlit dashboard
```

In the dashboard, open **Control**, choose the demo window, press **Start replay**, then watch **Live Weights**, **Signal Feed** and **Why Changed**.

Run tests:

```bash
pytest
```

Smoke test:

```bash
curl -X POST "localhost:8000/replay/start?speed=3000"
curl localhost:8000/weights/latest
```

**Notes**
- `make data` writes `data/prices.csv`, `data/raw/` and the signals file is generated by `make score`; these are git-ignored.
- If `data/signals.jsonl` or `data/prices.csv` is missing, the replay falls back to empty signals or flat dummy prices. Run `make data` and `make score` first.
- Replay errors are written to `replay_error.log` and shown by `/health`.
- Keep any keys in a local `.env` (git-ignored). Never commit keys.

| Symptom | Fix |
|---|---|
| Dashboard blank | Start the API first, press Start on Control, check `/health` |
| Weights chart flat | Use the demo window; check `data/signals.jsonl` has records for several tickers |
| yfinance empty | Retry later or use cached `data/prices.csv` |
| FinBERT download fails | Download once on a good connection; models are then cached |
| Port already in use | Close the old process or use another port |

## 5. Key Results & Domain Impact

### Backtest method

- Same `Rebalancer` class as the live replay.
- Decision at bar *i* uses only signals with `ts <= bar time` and earns the return from bar *i* to *i+1*.
- Metrics reported on the last ~40% of the timeline (the first 60% is warm-up).
- 5 bps cost charged on every rebalance, including baselines.
- Strategies: equal-weight hold; equal-weight daily rebalance; naive sentiment tilt (no decay, confidence or cost control); Frustrum (full); Frustrum with decay, dedup, contagion or cost control removed.
- Cost sensitivity at 0, 5, 10 and 25 bps for Frustrum vs the naive tilt.

> TODO: run `make backtest` and paste the real output below. Do not publish numbers this repo did not produce.

| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown | Annual turnover | Cost drag (bps/yr) | Tracking error | Mean IC |
|---|---|---|---|---|---|---|---|---|
| Equal-weight, hold | TODO | TODO | TODO | TODO | TODO | TODO | TODO | n/a |
| Equal-weight, daily rebal | TODO | TODO | TODO | TODO | TODO | TODO | TODO | n/a |
| Naive sentiment tilt | TODO | TODO | TODO | TODO | TODO | TODO | TODO | n/a |
| Frustrum (full) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Frustrum, no decay | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Frustrum, no dedup | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Frustrum, no contagion | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Frustrum, no cost control | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

All results come from **synthetic news** and real daily prices, so they validate the mechanics, not real-world alpha.


### Domain impact

Frustrum shows how a risk-aware desk could use text signals: as weak, noisy inputs that earn a small, bounded, costed position, with every move traceable to its headlines. It reduces manual news triage, surfaces event-driven risk, and enforces transparent weight limits. These are prototype capabilities, not evidence of production readiness or live-market alpha.

## Limitations

- **News is synthetic.** Backtest and validation results do not show that real news predicts returns.
- Impact is a rule-based score, not a model trained on historical price reactions. `scripts/train_impact.py` only prints the rule and base scores.
- Only daily bars are used. Sample covers 15 large-cap US stocks over about 2 years.
- Half-lives, sector table, fees, source credibility and expected return per unit of view are assumed values, not tuned on real data.
- FinBERT is English-only and can misread sarcasm.
- Simulation only: no market impact or slippage beyond the flat 5 bps fee.
- Past backtest performance does not predict future results.
