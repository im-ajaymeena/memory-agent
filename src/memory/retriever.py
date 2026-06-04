from datetime import datetime, timezone

import numpy as np

from .embedder import embed
from .models import MemoryRecord
from .store import MemoryStore
from .stub import Memory


def retrieve(query: str, store: MemoryStore, k: int = 8) -> list[Memory]:
    """
    Sync read path: embed query → numpy cosine top-K → Memory objects.
    Called inside asyncio.to_thread() in agent.py — stays synchronous intentionally.
    """
    query_vec = embed(query)
    records = store.search(query_vec, k=k)
    # Higher source_trust first, then most recent
    records.sort(key=lambda r: (r.source_trust, r.timestamp_updated), reverse=True)
    return [_to_memory(r) for r in records]


def _to_memory(record: MemoryRecord) -> Memory:
    return Memory(
        id=record.id,
        body=record.text,
        type=record.category.value,
        source=record.source.value,
        updated_at=datetime.fromtimestamp(
            record.timestamp_updated, tz=timezone.utc
        ).isoformat(),
        age_human_readable=_age(record.timestamp_updated),
    )


def _age(ts: float) -> str:
    delta = datetime.now(timezone.utc).timestamp() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)} minutes ago"
    if delta < 86400:
        return f"{int(delta / 3600)} hours ago"
    if delta < 604800:
        return f"{int(delta / 86400)} days ago"
    if delta < 2592000:
        return f"{int(delta / 604800)} weeks ago"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
