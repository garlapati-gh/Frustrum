from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel

class RawDoc(BaseModel):
    doc_id: str
    ts: datetime
    source: str
    kind: Literal["news", "social"]
    text: str
    url: Optional[str] = None
    domain: Optional[str] = None
    ticker_hint: Optional[str] = None

class Signal(BaseModel):
    signal_id: str
    doc_id: str
    ts: datetime
    ticker: str
    kind: str
    source: str
    text: str
    sentiment: float
    event_type: str
    impact: int
    confidence: float
    novelty: float
    url: Optional[str] = None
