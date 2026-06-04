import asyncio

import pytest

import src.extractor as extractor_module
from src.extractor import (
    _turns_since,
    drain_pending_extraction,
    init_extractor,
    schedule_extraction,
)
from src.session import Session, Turn


# ── helpers ──────────────────────────────────────────────────────────────────

class _CountingMemory:
    def __init__(self, delay: float = 0.0):
        self.calls: list[list] = []
        self._delay = delay

    def retrieve(self, query: str) -> list:
        return []

    async def extract_and_store(self, turns: list) -> None:
        await asyncio.sleep(self._delay)
        self.calls.append(list(turns))


def _session_with_turns(n: int, tmp_sessions) -> Session:
    s = Session()
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        s.append(Turn.now(role, f"msg {i}"))
    return s


# ── _turns_since ─────────────────────────────────────────────────────────────

def test_turns_since_none_returns_all(tmp_sessions):
    s = _session_with_turns(4, tmp_sessions)
    result = _turns_since(s, None)
    assert len(result) == 4


def test_turns_since_known_id_returns_tail(tmp_sessions):
    s = _session_with_turns(4, tmp_sessions)
    cursor_id = s.history[1].id
    result = _turns_since(s, cursor_id)
    assert len(result) == 2
    assert result[0].id == s.history[2].id


def test_turns_since_unknown_id_returns_all(tmp_sessions):
    s = _session_with_turns(3, tmp_sessions)
    result = _turns_since(s, "nonexistent-uuid")
    assert len(result) == 3


def test_turns_since_last_id_returns_empty(tmp_sessions):
    s = _session_with_turns(3, tmp_sessions)
    result = _turns_since(s, s.history[-1].id)
    assert result == []


# ── coalescing ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_coalescing_only_one_trailing_run(tmp_sessions):
    """
    5 rapid schedule_extraction calls while one is in-flight → at most 2 runs,
    not 5. The key property: no concurrent pile-up of extractions.

    With the same session (no new turns added between calls), the cursor advances
    after the first run so the trailing run finds nothing new and skips
    extract_and_store. Either 1 or 2 runs is correct; 5 is wrong.
    """
    mem = _CountingMemory(delay=0.05)
    extractor_module.set_memory(mem)

    s = _session_with_turns(5, tmp_sessions)

    for _ in range(5):
        schedule_extraction(s)

    await drain_pending_extraction(timeout_s=5.0)

    # Coalescing: never more than 2 runs regardless of how many rapid calls.
    assert len(mem.calls) <= 2
    assert len(mem.calls) >= 1


@pytest.mark.asyncio
async def test_coalescing_latest_session_used(tmp_sessions):
    """
    s2 is stashed, then s3 overwrites s2 in the stash.
    The trailing run should process s3's turns (3), not s2's turns (2).
    """
    mem = _CountingMemory(delay=0.05)
    extractor_module.set_memory(mem)

    s1 = _session_with_turns(1, tmp_sessions)
    s2 = _session_with_turns(2, tmp_sessions)
    s3 = _session_with_turns(3, tmp_sessions)

    schedule_extraction(s1)
    schedule_extraction(s2)  # stashed
    schedule_extraction(s3)  # overwrites s2 in stash

    await drain_pending_extraction(timeout_s=5.0)

    assert len(mem.calls) == 2
    # Second run processed s3 (3 turns), not s2 (2 turns).
    # Since s3's turn IDs differ from s1's, cursor miss → all 3 turns re-processed.
    assert len(mem.calls[1]) == 3


# ── cursor advancement ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cursor_advances_after_extraction(tmp_sessions):
    """Second extraction only processes the turns added since the first."""
    mem = _CountingMemory(delay=0.0)
    extractor_module.set_memory(mem)

    s = _session_with_turns(2, tmp_sessions)
    schedule_extraction(s)
    await drain_pending_extraction(timeout_s=5.0)
    first_call_len = len(mem.calls[0])

    # Add 1 more turn and extract again
    s.append(Turn.now("user", "new turn"))
    schedule_extraction(s)
    await drain_pending_extraction(timeout_s=5.0)
    second_call_len = len(mem.calls[1])

    assert first_call_len == 2
    assert second_call_len == 1  # only the new turn


# ── drain ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drain_waits_for_in_flight(tmp_sessions):
    """drain_pending_extraction returns only after the running task completes."""
    completed = []
    mem = _CountingMemory(delay=0.1)

    async def slow_extract(turns):
        await asyncio.sleep(0.1)
        completed.append(True)

    mem.extract_and_store = slow_extract  # type: ignore
    extractor_module.set_memory(mem)

    s = _session_with_turns(1, tmp_sessions)
    schedule_extraction(s)
    await drain_pending_extraction(timeout_s=5.0)

    assert completed == [True]


@pytest.mark.asyncio
async def test_drain_noop_when_nothing_in_flight():
    """drain_pending_extraction with empty _in_flight returns immediately."""
    await drain_pending_extraction(timeout_s=1.0)  # should not raise or hang
