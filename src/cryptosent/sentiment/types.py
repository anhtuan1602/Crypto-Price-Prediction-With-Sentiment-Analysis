from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class SentimentResult:
    score: float  # -1 (bearish) .. 1 (bullish)
    label: str  # manipulation | bullish | bearish | neutral
    confidence: float  # 0 .. 1
    reasoning: str = ""
    backend: str = ""
    model: str = ""

    def to_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["scored_at"] = datetime.now(timezone.utc).isoformat()
        return doc
