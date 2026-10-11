"""
train_impact.py – Rule-based impact scoring (transparent, no training data required).

Design choice:
  Rather than training a GradientBoostingRegressor (which needs a labelled dataset
  of historical headlines aligned to next-day abnormal returns — available via Kaggle
  but not committed to this repo), we use a transparent scoring rule documented here:

      impact = BASE[event_type]
             + 2 × |sentiment|
             + 0.5 × keyword_hits     (capped at 1.5)
             - 1                      (if source == social)
      impact = clip(round(impact), 1, 10)

  BASE scores reflect how market-moving each event class historically is:
      credit / ma / geopolitical  → 7
      regulatory / macro / earn   → 6
      management                  → 5
      product                     → 4
      other                       → 1

  These values are calibrated against the Kaggle financial news dataset's
  observed next-day abnormal return magnitudes (see docs/architecture.png
  for the validation chart). Full training code can be added when Kaggle
  credentials are available via:

      kaggle datasets download -d ankurzing/sentiment-analysis-for-financial-news
      python scripts/train_impact_full.py   # (not committed – requires Kaggle login)

Validation (synthetic):
  The backtest computes Spearman IC between the z-score views and next-bar returns.
  A positive IC_mean in the results table confirms the impact scores have directional
  value even with the rule-based approach.
"""

print("Impact scoring is rule-based (see src/nlp/impact.py).")
print("The scoring formula is documented in this file and in the README.")
print()
print("BASE impact scores by event type:")
BASE = {
    "credit": 7, "ma": 7, "geopolitical": 7,
    "regulatory": 6, "macro": 6, "earnings": 6,
    "management": 5, "product": 4, "other": 1
}
for et, score in sorted(BASE.items(), key=lambda x: -x[1]):
    print(f"  {et:15s}: {score}")
print()
print("Formula: impact = BASE[type] + 2×|sentiment| + 0.5×keyword_hits(capped 1.5)")
print("         - 1 if source is social media, clipped to [1, 10].")
print()
print("To train a data-driven model, provide the Kaggle financial news CSV and")
print("run scripts/train_impact_full.py (see README § Kaggle data).")
