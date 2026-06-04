"""
Tests mandated by the Architecture 14 testing strategy that were not yet covered.
Covers: metadata envelope completeness, update propagation, implicit staleness
        limitation, extractor coalescing, and timestamp invariants.
"""

import asyncio
import pathlib
import time

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.models import Category, MemoryRecord, Source
from src.memory.store import MemoryStore


# ── helpers ───────────────────────────────────────────────────────────────────

def _record(text: str = "User likes Go", **kwargs) -> MemoryRecord:
    return MemoryRecord(
        text=text,
        category=kwargs.get("category", Category.PREFERENCES),
        source=kwargs.get("source", Source.USER_STATEMENT),
        embedding=kwargs.get("embedding", [1.0, 0.0]),
        entities=kwargs.get("entities", ["Go"]),
        contextual_markers=kwargs.get("contextual_markers", ["work_context"]),
    )


@pytest.fixture
def store(tmp_path: pathlib.Path) -> MemoryStore:
    return MemoryStore(tmp_path / "spec.db")


# ═══════════════════════════════════════════════════════════════════════════════
# IMDMR metadata envelope completeness
# ═══════════════════════════════════════════════════════════════════════════════

def test_metadata_envelope_completeness(store: MemoryStore) -> None:
    """Every record written to the store must carry all IMDMR envelope fields."""
    r = _record()
    store.add(r)
    fetched = store.get_by_id(r.id)
    assert fetched is not None

    assert isinstance(fetched.text, str) and fetched.text
    assert isinstance(fetched.embedding, list) and len(fetched.embedding) > 0
    assert isinstance(fetched.entities, list)
    assert fetched.category in Category
    assert isinstance(fetched.intent_label, str)
    assert isinstance(fetched.contextual_markers, list)
    assert isinstance(fetched.timestamp_created, float) and fetched.timestamp_created > 0
    assert isinstance(fetched.timestamp_updated, float) and fetched.timestamp_updated > 0
    assert fetched.source in Source
    assert isinstance(fetched.is_current, bool)
    assert isinstance(fetched.source_trust, int) and fetched.source_trust in (1, 2, 3)


def test_contextual_markers_round_trip(store: MemoryStore) -> None:
    """contextual_markers survive the write → read round-trip unchanged."""
    markers = ["side_project", "deadline_q3", "python_context"]
    r = _record(contextual_markers=markers)
    store.add(r)
    fetched = store.get_by_id(r.id)
    assert fetched is not None
    assert fetched.contextual_markers == markers


def test_contextual_markers_empty_by_default(store: MemoryStore) -> None:
    """Records with no contextual_markers store and reload an empty list."""
    r = MemoryRecord(
        text="User uses vim",
        category=Category.PREFERENCES,
        source=Source.USER_STATEMENT,
        embedding=[0.5, 0.5],
    )
    store.add(r)
    fetched = store.get_by_id(r.id)
    assert fetched is not None
    assert fetched.contextual_markers == []


# ═══════════════════════════════════════════════════════════════════════════════
# UPDATE propagates to retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def test_update_propagates_to_retrieval(store: MemoryStore) -> None:
    """After an UPDATE the retriever sees the new text, not the original."""
    r = _record("User prefers tabs for indentation")
    store.add(r)

    store.update(r.id, "User prefers spaces, specifically 4-space indent", [0.9, 0.1])

    results = store.search(np.array([0.9, 0.1], dtype="float32"), k=1)
    assert len(results) == 1
    assert "spaces" in results[0].text
    assert "tabs" not in results[0].text


def test_updated_at_timestamp_bumped(store: MemoryStore) -> None:
    """timestamp_updated must be strictly greater than original after UPDATE."""
    r = _record()
    store.add(r)
    original_ts = r.timestamp_updated
    time.sleep(0.02)

    store.update(r.id, "User likes Go and Rust", [0.8, 0.2])
    fetched = store.get_by_id(r.id)
    assert fetched is not None
    assert fetched.timestamp_updated > original_ts
    assert fetched.timestamp_created == pytest.approx(r.timestamp_created, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# Implicit staleness is NOT detected (documents known limitation)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_implicit_staleness_not_detected(store: MemoryStore) -> None:
    """
    Known limitation: if a fact becomes stale without an explicit contradiction,
    the old record persists indefinitely. This test documents the gap, not a bug.

    Scenario: user says "I prefer Python" then later "I've been writing a lot of Rust"
    without explicitly saying "I switched from Python". The Python preference is
    NOT removed — both records coexist.
    """
    from src.memory.adjudicator import adjudicate
    from src.memory.models import CandidateFact

    store.add(MemoryRecord(
        text="User prefers Python for scripting",
        category=Category.PREFERENCES,
        source=Source.USER_STATEMENT,
        embedding=[1.0, 0.0],
    ))

    # Drift without explicit contradiction — "I've been writing Rust lately"
    drifted = CandidateFact(
        text="User has been writing a lot of Rust lately",
        category=Category.PREFERENCES,
        source=Source.USER_STATEMENT,
    )

    with patch("src.memory.adjudicator._client") as mock_llm, \
         patch("src.memory.adjudicator.embed") as mock_embed:
        mock_embed.return_value = np.array([0.4, 0.6], dtype="float32")
        import json
        msg = MagicMock()
        msg.content = [MagicMock(text=json.dumps({"operation": "ADD", "target_id": None}))]
        mock_llm.messages.create = AsyncMock(return_value=msg)
        await adjudicate(drifted, store)

    # Both records coexist — implicit staleness is NOT resolved
    active = store.all_active()
    assert len(active) == 2
    texts = {r.text for r in active}
    assert any("Python" in t for t in texts), "Python preference still present"
    assert any("Rust" in t for t in texts), "Rust fact also stored"


# ═══════════════════════════════════════════════════════════════════════════════
# Extractor coalescing — rapid turns are not lost
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_rapid_turns_queue_correctly(tmp_sessions) -> None:
    """
    10 schedule_extraction calls fired in rapid succession must not lose turns.
    Coalescing means at most 2 extraction runs occur; all new turns are processed.
    """
    import src.extractor as ext
    from src.session import Session, Turn

    ext.init_extractor()

    processed_turns: list[list] = []

    async def fake_extract(turns):
        processed_turns.append(list(turns))

    class FakeMemory:
        async def extract_and_store(self, turns):
            await fake_extract(turns)

    ext.set_memory(FakeMemory())

    session = Session()
    for i in range(10):
        session.append(Turn.now("user", f"message {i}"))
        session.append(Turn.now("assistant", f"reply {i}"))
        ext.schedule_extraction(session)

    # Drain all in-flight tasks
    await ext.drain_pending_extraction(timeout_s=10.0)

    # All 20 turns (10 user + 10 assistant) must have been processed
    total_processed = sum(len(batch) for batch in processed_turns)
    all_turns_in_session = len(session.history)
    assert total_processed == all_turns_in_session, (
        f"Expected {all_turns_in_session} turns processed, got {total_processed}"
    )

    # Clean up
    ext.init_extractor()
    ext.set_memory(None)


# ═══════════════════════════════════════════════════════════════════════════════
# Soft-delete audit trail
# ═══════════════════════════════════════════════════════════════════════════════

def test_soft_deleted_record_excluded_from_retrieval_but_on_disk(
    store: MemoryStore,
) -> None:
    """Soft-deleted record must not surface in search but must survive on disk."""
    r = _record(embedding=[1.0, 0.0])
    store.add(r)
    store.soft_delete(r.id)

    results = store.search(np.array([1.0, 0.0], dtype="float32"), k=10)
    assert all(rec.id != r.id for rec in results), "deleted record appeared in search"

    ghost = store.get_by_id(r.id)
    assert ghost is not None
    assert ghost.is_current is False
