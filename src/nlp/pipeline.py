from src.nlp.entity import link_tickers
from src.nlp.sentiment import score_sentiment
from src.nlp.events import embed, classify_event
from src.nlp.dedup import novelty
from src.nlp.impact import predict_impact
from src.common.schemas import Signal

SRC_CRED = {
    "gdelt": 0.8, "newsapi": 0.8, "tweets": 0.5, "reddit": 0.4,
    "synthetic_news": 0.8, "synthetic_social": 0.5
}

def process(docs):
    docs = sorted(docs, key=lambda d: d.ts)
    pairs = [(d, t) for d in docs for t in (link_tickers(d) or ["*"])]
    texts = [d.text for d, _ in pairs]
    if not texts: return []

    vecs = embed(texts)
    sent = score_sentiment(texts)
    ev = classify_event(texts, vecs)

    signals = []
    for (d, t), (s, mconf), (e, econf), v in zip(pairs, sent, ev, vecs):
        nov = novelty.check(d.ts, t, v)
        conf = 0.4 * mconf + 0.3 * econf + 0.3 * SRC_CRED.get(d.source, 0.5)
        imp = predict_impact(e, s, d.text, d.kind, nov)
        signals.append(Signal(
            signal_id=f"{d.doc_id}:{t}", doc_id=d.doc_id, ts=d.ts,
            ticker=t, kind=d.kind, source=d.source, text=d.text,
            sentiment=round(s, 4), event_type=e, impact=imp,
            confidence=round(conf, 3), novelty=round(nov, 3), url=d.url
        ))
    return signals
