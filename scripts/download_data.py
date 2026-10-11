"""
Generate synthetic demo data spanning 2 years of daily price history.
NOTE: These signals are synthetic for demo purposes, calibrated to match realistic event distributions.
"""
import os
import json
import random
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
from src.ingest.prices import load_prices


TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "WMT", "KO", "BA", "DIS"]

TEMPLATES = {
    "earnings": [
        "{t} reports record quarterly net income of ${eps}B, surging {pct}% year-over-year above consensus.",
        "{t} misses earnings expectations significantly as Q{q} operating margins compress by {pct}%.",
        "{t} raises full-year profit and EPS guidance following unprecedented enterprise demand.",
        "{t} warns of severe revenue headwinds in upcoming quarter, cutting guidance by {pct}%.",
        "{t} posts stellar quarterly sales across all operational divisions, beating analyst targets.",
        "{t} reports surprising net loss driven by unforeseen restructuring charges and weaker orders."
    ],
    "credit": [
        "Major credit rating agency upgrades {t} senior debt rating to AA+ with stable outlook.",
        "S&P downgrades {t} corporate credit rating to junk status following debt surge.",
        "{t} successfully refinances ${val}B debt portfolio at substantially lower interest rates.",
        "{t} faces liquidity concerns and elevated default spreads as short-term bond maturities loom.",
        "{t} issues ${val}B investment-grade bonds with overwhelming investor oversubscription.",
        "Credit analysts warn {t} debt-to-EBITDA ratio has reached dangerous multi-year highs."
    ],
    "ma": [
        "{t} announces definitive agreement to acquire industry innovator in all-cash ${val}B transaction.",
        "{t} proposed mega-merger officially called off due to insurmountable antitrust objections.",
        "{t} completes accretive strategic buyout expected to add ${val}M in annual synergies.",
        "{t} terminates acquisition discussions after audit reveals undisclosed contingent liabilities.",
        "{t} confirms hostile takeover bid rejected by shareholders, preserving independent operations.",
        "{t} announces joint venture partnership expanding distribution across global markets."
    ],
    "regulatory": [
        "Federal court dismisses major antitrust allegations against {t}, ending multi-year investigation.",
        "Regulatory authority launches formal monopoly probe into {t} business practices.",
        "{t} granted pivotal regulatory clearance and key patent protections across all jurisdictions.",
        "Government body imposes unprecedented ${val}M regulatory fine on {t} for compliance failures.",
        "Regulators approve {t} standard compliance roadmap without imposing structural remedies.",
        "{t} faces Congressional inquiry and potential restrictive legislation regarding commercial dominance."
    ],
    "geopolitical": [
        "Bilateral trade pact lowers international tariffs on {t} export products by {pct}%.",
        "Geopolitical sanctions abruptly halt {t} cross-border operations and freeze supply shipments.",
        "{t} successfully diversifies manufacturing infrastructure away from regional conflict zones.",
        "New retaliatory import tariffs threaten ${val}B in annual foreign revenues for {t}.",
        "International trade committee clears {t} supply network of unfair subsidization claims.",
        "Critical regional transport route blockade delays key industrial components destined for {t}."
    ],
    "management": [
        "{t} board appoints universally acclaimed industry leader as new Chief Executive Officer.",
        "{t} CEO abruptly steps down following deep strategic disagreements with executive board.",
        "{t} leadership team outlines visionary multi-year shareholder return and operational roadmap.",
        "{t} suffers sudden wave of C-suite resignations raising succession and leadership concerns.",
        "{t} announces comprehensive corporate restructuring expected to optimize overhead efficiency.",
        "Unionized labor workforce at {t} critical production facilities votes to initiate indefinite strike."
    ],
    "product": [
        "{t} unveils groundbreaking next-generation platform featuring industry-first capabilities.",
        "{t} announces urgent worldwide product recall of {val}M units due to critical safety defect.",
        "{t} flagship product rollout achieves record-breaking initial consumer adoption figures.",
        "{t} indefinitely postpones highly anticipated product release citing severe quality control setbacks.",
        "{t} expands commercial availability of revolutionary product line into ten new international markets.",
        "Early benchmark reviews criticize {t} latest hardware revision for severe performance bottlenecks."
    ],
    "macro": [
        "Federal Reserve signals dovish monetary easing cycle, driving broad equity valuation expansion.",
        "Central bank executes aggressive 50 bps interest rate hike to combat stubbornly persistent inflation.",
        "Robust national consumer spending and GDP data fuel economic optimism across industrial sectors.",
        "Consumer sentiment index plummets to historic low amid mounting stagflation worries.",
        "Benchmark bond yields retreat significantly as economic soft-landing indicators strengthen.",
        "Global manufacturing PMI contracts unexpectedly, signaling broader industrial slowdown."
    ]
}

SOURCE_POOL = ["synthetic_news", "synthetic_social", "synthetic_news", "synthetic_news", "synthetic_social"]
KIND_POOL   = ["news", "social", "news", "news", "social"]


def make_doc(doc_id, ts, ticker, etype, tmpl_idx, variant):
    tmpl = TEMPLATES[etype][tmpl_idx % len(TEMPLATES[etype])]
    text = tmpl.format(
        t=ticker,
        pct=random.randint(4, 28),
        eps=round(random.uniform(1.2, 9.5), 2),
        val=random.randint(2, 65),
        q=random.randint(1, 4),
    )
    # Ensure distinct phrasing variation
    if variant > 0:
        text = text + f" (Report update #{variant})"
    src_idx = random.randint(0, len(SOURCE_POOL) - 1)
    return {
        "doc_id": doc_id,
        "ts": ts.isoformat(),
        "source": SOURCE_POOL[src_idx],
        "kind": KIND_POOL[src_idx],
        "text": text,
        "ticker_hint": ticker,
        "event_hint": etype,
    }


def main():
    print("Downloading 2 years of daily prices from yfinance...")
    df = load_prices(TICKERS, period="2y", interval="1d")
    df.to_csv("data/prices.csv")
    print(f"Prices saved: {df.shape[0]} days x {df.shape[1]} tickers")

    df.tail(30).to_csv("data/prices_sample.csv")

    start_dt = df.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
    end_dt   = df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    date_range_days = (end_dt - start_dt).days
    print(f"Price range: {start_dt.date()} to {end_dt.date()} ({date_range_days} days)")

    print("Generating synthetic news events across 2-year price range...")
    random.seed(42)
    np.random.seed(42)

    docs = []
    etypes = list(TEMPLATES.keys())
    doc_counter = 0

    # Ensure demo window is set in early 2026 where dense events occur
    demo_start = end_dt - timedelta(days=20)
    demo_end = end_dt - timedelta(days=6)

    # 1. Background events spread across the 2 years
    for ticker in TICKERS:
        for etype in etypes:
            for k in range(4):
                offset_days = random.randint(30, date_range_days - 25)
                offset_hours = random.randint(8, 20)
                ts = start_dt + timedelta(days=offset_days, hours=offset_hours)
                tmpl_idx = k % len(TEMPLATES[etype])
                docs.append(make_doc(f"syn_{doc_counter}", ts, ticker, etype, tmpl_idx, variant=0))
                doc_counter += 1

    # 2. Dense demo window events (14-day window: demo_start to demo_end)
    demo_days = (demo_end - demo_start).days
    for ticker in TICKERS:
        for etype in ["earnings", "product", "management", "regulatory"]:
            offset_d = random.randint(1, demo_days - 1)
            ts = demo_start + timedelta(days=offset_d, hours=random.randint(9, 17))
            docs.append(make_doc(f"demo_{doc_counter}", ts, ticker, etype, random.randint(0, 5), variant=1))
            doc_counter += 1

    # 3. Planted circuit-breaker event for BA in the demo window!
    # A severe fraud & regulatory probe with massive negative sentiment and impact 10
    cb_ts = demo_start + timedelta(days=5, hours=14, minutes=30)
    docs.append({
        "doc_id": "cb_ba_fraud",
        "ts": cb_ts.isoformat(),
        "source": "synthetic_news",
        "kind": "news",
        "text": "Boeing faces catastrophic fraud probe: SEC halts trading over massive balance sheet irregularities worth billions.",
        "ticker_hint": "BA",
        "event_hint": "regulatory",
    })

    # 4. Macro events
    for k in range(12):
        offset_days = random.randint(30, date_range_days - 5)
        ts = start_dt + timedelta(days=offset_days, hours=15)
        macro_text = random.choice([
            "Federal Reserve signals unexpected interest rate tightening amid stubborn wage data.",
            "Federal Reserve announces interest rate cut cycle as inflation returns to target.",
            "National GDP expansion exceeds expectations, boosting broad equity sentiment.",
            "Global trade volume contracts sharply, raising industrial macroeconomic concerns."
        ])
        docs.append({
            "doc_id": f"macro_{k}",
            "ts": ts.isoformat(),
            "source": "synthetic_news",
            "kind": "news",
            "text": macro_text,
            "ticker_hint": "*",
            "event_hint": "macro",
        })

    docs.sort(key=lambda d: d["ts"])

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/sample_news.jsonl"
    with open(out_path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    sample_path = "data/sample_news.jsonl"
    with open(sample_path, "w") as f:
        for d in docs[:200]:
            f.write(json.dumps(d) + "\n")

    # Update demo_events.json to point exactly to this 14-day window!
    demo_events = [
        {
            "name": "Live Demo Window (2 Weeks + BA Circuit Breaker)",
            "start": demo_start.isoformat(),
            "end": demo_end.isoformat()
        }
    ]
    with open("data/demo_events.json", "w") as f:
        json.dump(demo_events, f, indent=2)

    print(f"Generated {len(docs)} synthetic events -> {out_path}")
    print(f"Configured demo window in data/demo_events.json: {demo_start.date()} to {demo_end.date()}")


if __name__ == "__main__":
    main()
