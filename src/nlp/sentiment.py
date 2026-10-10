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
