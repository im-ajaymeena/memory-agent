import json

import pytest

from src.session import Session, Turn


def test_turn_now_generates_id_and_timestamp():
    t = Turn.now("user", "hello")
    assert t.role == "user"
    assert t.content == "hello"
    assert t.id != ""
    assert "T" in t.timestamp  # ISO 8601


def test_session_append_writes_to_disk(tmp_sessions):
    s = Session()
    t = Turn.now("user", "hi")
    s.append(t)

    assert s.path.exists()
    lines = s.path.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["content"] == "hi"
    assert data["role"] == "user"
    assert data["id"] == t.id


def test_session_load_replays_history(tmp_sessions):
    s1 = Session()
    s1.append(Turn.now("user", "first"))
    s1.append(Turn.now("assistant", "second"))

    s2 = Session(s1.session_id).load()
    assert len(s2.history) == 2
    assert s2.history[0].content == "first"
    assert s2.history[1].content == "second"


def test_session_load_empty_file(tmp_sessions):
    s = Session()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    s.path.write_text("")
    s2 = Session(s.session_id).load()
    assert s2.history == []


def test_session_survives_restart(tmp_sessions):
    """Simulates process restart: write in one Session object, load in another."""
    sid = None
    s = Session()
    sid = s.session_id
    for i in range(5):
        s.append(Turn.now("user", f"msg {i}"))

    reloaded = Session(sid).load()
    assert len(reloaded.history) == 5
    assert reloaded.history[4].content == "msg 4"


def test_last_n_turns(tmp_sessions):
    s = Session()
    for i in range(10):
        s.append(Turn.now("user", f"msg {i}"))
    last3 = s.last_n_turns(3)
    assert len(last3) == 3
    assert last3[-1].content == "msg 9"


def test_list_all_returns_sessions(tmp_sessions):
    s1 = Session()
    s1.append(Turn.now("user", "a"))
    s2 = Session()
    s2.append(Turn.now("user", "b"))

    sessions = Session.list_all()
    ids = [s["id"] for s in sessions]
    assert s1.session_id in ids
    assert s2.session_id in ids
