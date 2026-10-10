# IIITL-Gagana-Hackathon

Demo Video: // youtubee
Slide Deck: //docs,presentation

## 1. Overview and approach
Frustrum is an event-aware, cost-aware rebalancer. It treats news NLP output as noisy views about stocks, turning those views into trades with confidence, decay, costs, and limits. We maintain a benchmark index and let the NLP engine push weights away from it only as far as evidence justifies. 

## 2. Architecture and tech stack
![Architecture](docs/architecture.png)

## 3. Dataset Used
- **GDELT DOC 2.0 API**: Live news per ticker.
- **Kaggle stock tweets**: Replayed social stream.
- **Kaggle financial news headlines**: Historical news for impact model.
- **yfinance**: Prices, returns, and covariance.

**Assumptions**: 
- Half-lives by event type are hand-set starting points.
- Sector sensitivities for macro/geopolitical events.
- 5 bps transaction cost.
- Source credibility values: news=1.0, social=0.5.
- Expected return per unit of z-score view is 10 bps.

## 4. Quickstart
1. `make setup` - Install dependencies
2. `make score` - Precompute NLP scores
3. `make run-api` - Start the FastAPI backend (in one terminal)
4. `make dashboard` - Start the Streamlit dashboard (in another terminal)

## 5. Key results and domain impact
We evaluate against equal-weight, equal-weight daily rebalanced, and naive sentiment tilts. By accounting for transaction costs, turnover limits, and decaying news impact, Frustrum avoids chasing noise and limits drawdown.

## Limitations
The backtest sample is short, limited to 15 large-cap stocks. NLP signals from FinBERT are noisy and can misread sarcasm. The current version is a simulation with fixed slippage (5 bps) and no market impact modeled.

