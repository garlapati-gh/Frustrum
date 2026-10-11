BASE = {"credit": 7, "ma": 7, "geopolitical": 7, "regulatory": 6, "macro": 6, "earnings": 6, "management": 5, "product": 4, "other": 1}
CUES = ["billion", "recall", "probe", "fraud", "bankruptcy", "halt", "ban", "guidance"]

def predict_impact(event, sentiment, text, kind, nov=1.0) -> int:
    s = BASE.get(event, 1) + 2 * abs(sentiment)
    # Severe cues give stronger boost
    t_lower = text.lower()
    if any(k in t_lower for k in ["fraud", "bankruptcy", "halt"]):
        s += 2.0
    elif any(k in t_lower for k in CUES):
        s += 1.0
    s -= 1 if kind == "social" else 0
    return int(max(1, min(10, round(s))))
