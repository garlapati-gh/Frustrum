import os
import textwrap

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent(content).strip() + '\n')

create_file('src/nlp/entity.py', '''
import re
import pandas as pd

_PAT = {}

def load_tickers():
    try:
        T = pd.read_csv("data/tickers.csv")
        for _, r in T.iterrows():
            names = [r["name"]] + [a for a in str(r["aliases"]).split("|") if a and a != "nan"]
            body = "|".join(re.escape(n) for n in names)
            _PAT[r["ticker"]] = re.compile(rf"\\b({body})\\b|\\${r['ticker']}\\b", re.I)
    except FileNotFoundError:
        pass

def link_tickers(doc) -> list[str]:
    if not _PAT: load_tickers()
    if doc.ticker_hint:
        return [doc.ticker_hint]
    return [t for t, p in _PAT.items() if p.search(doc.text)]
''')

create_file('src/nlp/sentiment.py', '''
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_tok = None
_model = None
_LABELS = None

def init_model():
    global _tok, _model, _LABELS
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
        _LABELS = [_model.config.id2label[i].lower() for i in range(3)]

@torch.no_grad()
def score_sentiment(texts: list[str], batch: int = 32) -> list[tuple[float, float]]:
    init_model()
    out = []
    for i in range(0, len(texts), batch):
        enc = _tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
        probs = torch.softmax(_model(**enc).logits, dim=-1)
        for row in probs:
            d = dict(zip(_LABELS, row.tolist()))
            out.append((d["positive"] - d["negative"], max(d.values())))
    return out
''')

create_file('src/nlp/events.py', '''
import json
from sentence_transformers import SentenceTransformer

_emb = None
_PROTOS = None
_LAB = None
_TXT = None
_P = None

KEYWORDS = {
    "credit": ["downgrade", "default", "junk", "bankruptcy"],
    "ma": ["acquire", "merger", "takeover", "buyout"],
    "earnings": ["earnings", "quarterly results", "eps", "guidance"],
    "regulatory": ["antitrust", "sec probe", "fine", "lawsuit"],
    "geopolitical": ["sanction", "tariff", "war", "embargo"]
}

def init_embedder():
    global _emb, _PROTOS, _LAB, _TXT, _P
    if _emb is None:
        _emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        try:
            _PROTOS = json.load(open("data/event_prototypes.json"))
            _LAB, _TXT = zip(*[(k, s) for k, v in _PROTOS.items() for s in v])
            _P = _emb.encode(list(_TXT), normalize_embeddings=True)
        except:
            _LAB, _TXT, _P = [], [], None

def embed(texts):
    init_embedder()
    return _emb.encode(texts, normalize_embeddings=True, batch_size=64)

def classify_event(texts, vecs=None):
    init_embedder()
    vecs = embed(texts) if vecs is None else vecs
    out = []
    if _P is None:
        return [("other", 0.0) for _ in texts]
    
    for text, sims in zip(texts, vecs @ _P.T):
        score = {}
        for lab, v in zip(_LAB, sims):
            score[lab] = max(score.get(lab, -1.0), float(v))
        low = text.lower()
        for lab, words in KEYWORDS.items():
            if any(w in low for w in words):
                score[lab] = score.get(lab, 0.0) + 0.10
        (lab, top), (_, second) = sorted(score.items(), key=lambda kv: -kv[1])[:2]
        out.append(("other" if top < 0.25 else lab, min(1.0, max(0.0, (top - second) * 4))))
    return out
''')

create_file('src/nlp/dedup.py', '''
from datetime import timedelta

class NoveltyIndex:
    def __init__(self, window_hours: int = 24, thr: float = 0.88):
        self.window = timedelta(hours=window_hours)
        self.thr = thr
        self.items = []

    def check(self, ts, ticker, vec) -> float:
        self.items = [x for x in self.items if x[0] >= ts - self.window]
        top = max((float(vec @ v) for _, t, v in self.items if t == ticker), default=0.0)
        self.items.append((ts, ticker, vec))
        return 0.0 if top >= self.thr else 1.0 - top

novelty = NoveltyIndex()
''')

create_file('src/nlp/impact.py', '''
BASE = {"credit": 7, "ma": 7, "geopolitical": 7, "regulatory": 6, "macro": 6, "earnings": 6, "management": 5, "product": 4, "other": 1}
CUES = ["billion", "recall", "probe", "fraud", "bankruptcy", "halt", "ban", "guidance"]

def predict_impact(event, sentiment, text, kind, nov=1.0) -> int:
    s = BASE.get(event, 1) - 1 + 2 * abs(sentiment)
    s += min(1.5, 0.5 * sum(c in text.lower() for c in CUES))
    s -= 1 if kind == "social" else 0
    return int(max(1, min(10, round(s))))
''')

create_file('src/nlp/pipeline.py', '''
from src.nlp.entity import link_tickers
from src.nlp.sentiment import score_sentiment
from src.nlp.events import embed, classify_event
from src.nlp.dedup import novelty
from src.nlp.impact import predict_impact
from src.common.schemas import Signal

SRC_CRED = {"gdelt": 0.8, "newsapi": 0.8, "tweets": 0.5, "reddit": 0.4}

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
''')

