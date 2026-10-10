import hashlib
from datetime import datetime, timezone
import requests
from src.common.schemas import RawDoc

API = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_gdelt(query: str, timespan: str = "1d", maxrecords: int = 100) -> list[RawDoc]:
    params = {"query": f"({query}) sourcelang:english", "mode": "artlist", "format": "json", "maxrecords": maxrecords, "timespan": timespan, "sort": "datedesc"}
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    try:
        articles = r.json().get("articles", [])
    except ValueError:
        return []
    docs = []
    for a in articles:
        ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        docs.append(RawDoc(doc_id=hashlib.md5(a["url"].encode()).hexdigest(), ts=ts, source="gdelt", kind="news", text=a["title"], url=a["url"], domain=a.get("domain")))
    return docs
