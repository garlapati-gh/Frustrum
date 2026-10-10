import json, time
from pathlib import Path
from src.common.schemas import RawDoc

def load_docs(paths) -> list[RawDoc]:
    docs = [RawDoc(**json.loads(line)) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    return sorted(docs, key=lambda d: d.ts)

def replay(docs, speed: float = 600.0, instant: bool = False):
    prev = None
    for d in docs:
        if prev is not None and not instant:
            time.sleep(max(0.0, (d.ts - prev).total_seconds() / speed))
        prev = d.ts
        yield d
