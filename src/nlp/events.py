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
