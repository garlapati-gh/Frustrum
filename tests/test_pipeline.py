from datetime import datetime, timezone, timedelta
import pytest
from src.common.schemas import RawDoc
from src.nlp.pipeline import process

def test_pipeline_nlp_stages():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    docs = [
        # 1. Downgrade (credit)
        RawDoc(
            doc_id="doc_downgrade",
            ts=now,
            source="newsapi",
            kind="news",
            text="Moody's slashes Boeing credit rating to junk due to persistent debt problems.",
            ticker_hint="BA"
        ),
        # 2. Merger (ma)
        RawDoc(
            doc_id="doc_merger",
            ts=now + timedelta(minutes=5),
            source="newsapi",
            kind="news",
            text="ExxonMobil agrees to acquire Pioneer Natural Resources in mega oil merger.",
            ticker_hint="XOM"
        ),
        # 3. Product launch (product)
        RawDoc(
            doc_id="doc_product",
            ts=now + timedelta(minutes=10),
            source="newsapi",
            kind="news",
            text="Apple unveils breakthrough AI iPhone with revolutionary next-generation chips.",
            ticker_hint="AAPL"
        ),
        # 4. Duplicate story (same/near-identical text as #3, should get novelty 0)
        RawDoc(
            doc_id="doc_dup",
            ts=now + timedelta(minutes=15),
            source="newsapi",
            kind="news",
            text="Apple unveils breakthrough AI iPhone with revolutionary next-generation chips.",
            ticker_hint="AAPL"
        ),
        # 5. Macro story (macro)
        RawDoc(
            doc_id="doc_macro",
            ts=now + timedelta(minutes=20),
            source="newsapi",
            kind="news",
            text="Federal Reserve announces surprise interest rate hike as inflation stays elevated.",
            ticker_hint="*"
        ),
    ]

    signals = process(docs)
    assert len(signals) >= 5

    sig_by_id = {s.doc_id: s for s in signals}
    
    # 1. Downgrade
    s_down = sig_by_id["doc_downgrade"]
    assert s_down.event_type == "credit"
    assert s_down.sentiment < 0  # Negative sentiment

    # 2. Merger
    s_ma = sig_by_id["doc_merger"]
    assert s_ma.event_type == "ma"

    # 3. Product
    s_prod = sig_by_id["doc_product"]
    assert s_prod.event_type == "product"
    assert s_prod.sentiment > 0  # Positive sentiment

    # 4. Duplicate
    s_dup = sig_by_id["doc_dup"]
    assert s_dup.novelty == 0.0  # Duplicate gets 0 novelty

    # 5. Macro
    s_macro = sig_by_id["doc_macro"]
    assert s_macro.event_type == "macro"

