import pytest

from src.agent import MAX_TURNS_VERBATIM, _build_messages
from src.session import Session, Turn


def _make_session(n_turns: int, tmp_sessions) -> Session:
    s = Session()
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        s.append(Turn.now(role, f"turn {i}"))
    return s


def test_short_history_all_included(tmp_sessions):
    s = _make_session(4, tmp_sessions)
    msgs = _build_messages(s, "new question")
    # 4 history turns + 1 new user message
    assert len(msgs) == 5
    assert msgs[-1] == {"role": "user", "content": "new question"}


def test_long_history_truncated_to_max(tmp_sessions):
    s = _make_session(MAX_TURNS_VERBATIM + 10, tmp_sessions)
    msgs = _build_messages(s, "new question")
    # MAX_TURNS_VERBATIM history turns + 1 new user message
    assert len(msgs) == MAX_TURNS_VERBATIM + 1


def test_empty_history(tmp_sessions):
    s = Session()
    msgs = _build_messages(s, "hello")
    assert len(msgs) == 1
    assert msgs[0] == {"role": "user", "content": "hello"}


def test_newest_turns_preserved(tmp_sessions):
    s = _make_session(MAX_TURNS_VERBATIM + 5, tmp_sessions)
    msgs = _build_messages(s, "q")
    # The second-to-last message in history should be the most recent assistant turn
    second_last = msgs[-2]
    assert "turn" in second_last["content"]
    # The last turn index is MAX_TURNS_VERBATIM + 4 (0-based)
    # After truncation, we keep last MAX_TURNS_VERBATIM, so last history turn is index -1
    last_history_turn_idx = MAX_TURNS_VERBATIM + 5 - 1
    assert msgs[-2]["content"] == f"turn {last_history_turn_idx}"
