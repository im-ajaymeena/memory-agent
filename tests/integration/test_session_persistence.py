"""
Integration tests: session survives a simulated process restart.
No LLM calls — tests real JSONL I/O.
"""
import pytest

from src.session import Session, Turn


def test_session_persists_across_object_recreation(tmp_sessions):
    sid = None

    # Simulate first process: write turns
    s1 = Session()
    sid = s1.session_id
    s1.append(Turn.now("user", "I prefer Python"))
    s1.append(Turn.now("assistant", "Got it, noted."))
    s1.append(Turn.now("user", "I work at a fintech company"))

    # Simulate restart: create new Session object from same ID
    s2 = Session(sid).load()

    assert len(s2.history) == 3
    assert s2.history[0].content == "I prefer Python"
    assert s2.history[2].content == "I work at a fintech company"


def test_session_ids_preserved_after_reload(tmp_sessions):
    s1 = Session()
    t = Turn.now("user", "hello")
    s1.append(t)

    s2 = Session(s1.session_id).load()
    assert s2.history[0].id == t.id  # cursor UUIDs survive disk round-trip


def test_partial_write_does_not_corrupt(tmp_sessions):
    """Append valid lines, then a blank, then more valid lines — all parse."""
    s = Session()
    s.append(Turn.now("user", "line 1"))
    s.append(Turn.now("assistant", "line 2"))

    # Inject a blank line (simulates a partial write)
    with open(s.path, "a") as f:
        f.write("\n")

    s.append(Turn.now("user", "line 3"))

    reloaded = Session(s.session_id).load()
    assert len(reloaded.history) == 3


def test_multiple_sessions_independent(tmp_sessions):
    sa = Session()
    sb = Session()
    sa.append(Turn.now("user", "session A"))
    sb.append(Turn.now("user", "session B"))

    ra = Session(sa.session_id).load()
    rb = Session(sb.session_id).load()

    assert ra.history[0].content == "session A"
    assert rb.history[0].content == "session B"
    assert ra.session_id != rb.session_id
