"""
E2E tests — CLI mode (repl()) with real LLM calls.

Exercises the full CLI stack: session init, multi-turn conversation,
/clear, /sessions, session resume, and drain-on-exit.

Input is injected by patching PromptSession.prompt_async so no terminal is
needed. All session I/O goes to tmp_sessions via the conftest fixture.

Requires: ANTHROPIC_API_KEY
Run with: make test-e2e
"""
import asyncio
from collections import deque
from unittest.mock import patch, AsyncMock

import pytest

import src.agent as agent_module
from src.cli import repl
from src.extractor import init_extractor
from src.memory.stub import VanillaMemory
from src.session import Session, Turn


pytestmark = pytest.mark.slow


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_prompt(inputs: list[str]):
    """
    Returns an async callable that yields inputs one by one, then raises
    EOFError to terminate the REPL cleanly (same as ctrl-D).
    """
    q = deque(inputs)

    async def _prompt_async(*args, **kwargs):
        if q:
            return q.popleft()
        raise EOFError

    return _prompt_async


def _patch_prompt(inputs: list[str]):
    return patch("src.cli.PromptSession.prompt_async", new=_fake_prompt(inputs))


def _latest_session() -> Session:
    """Return the most-recently-modified session on disk."""
    sessions = Session.list_all()
    assert sessions, "No sessions found on disk"
    return Session(sessions[0]["id"]).load()


# ── multi-turn in one CLI run ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cli_multi_turn_persisted(tmp_sessions):
    """
    Two user messages in one REPL run — both turns are written to disk.
    """
    with _patch_prompt([
        "My name is Jordan.",
        "What is my name?",
    ]):
        await repl(session_id=None)

    s = _latest_session()
    assert len(s.history) == 4  # 2 user + 2 assistant

    user_msgs = [t.content for t in s.history if t.role == "user"]
    assert "My name is Jordan." in user_msgs
    assert "What is my name?" in user_msgs


@pytest.mark.asyncio
async def test_cli_second_turn_references_first(tmp_sessions):
    """
    History is sent to the API: the agent's second response must reference
    what was stated in the first turn.
    """
    with _patch_prompt([
        "My favourite animal is a capybara.",
        "What animal did I just mention?",
    ]):
        await repl(session_id=None)

    s = _latest_session()
    assistant_replies = [t.content for t in s.history if t.role == "assistant"]
    assert len(assistant_replies) == 2
    assert "capybara" in assistant_replies[1].lower(), (
        f"Expected 'capybara' in second reply, got: {assistant_replies[1]}"
    )


# ── session resume ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cli_resume_continues_history(tmp_sessions):
    """
    Start a session, exit, then resume it — the resumed run must see prior
    turns and the conversation must continue coherently.
    """
    # Run 1: establish a fact
    with _patch_prompt(["My project is called Nighthawk."]):
        await repl(session_id=None)

    sid = _latest_session().session_id
    init_extractor()

    # Run 2: resume same session and ask about the fact
    with _patch_prompt(["What is my project called?"]):
        await repl(session_id=sid)

    reloaded = Session(sid).load()
    # Run 1: 2 turns. Run 2: 2 more turns.
    assert len(reloaded.history) == 4

    last_assistant = [t for t in reloaded.history if t.role == "assistant"][-1]
    assert "nighthawk" in last_assistant.content.lower(), (
        f"Expected agent to recall 'Nighthawk', got: {last_assistant.content}"
    )


# ── /clear command ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cli_clear_creates_new_session(tmp_sessions):
    """
    /clear in the middle of a run must start a fresh session — two distinct
    session IDs must exist on disk after the run.
    """
    with _patch_prompt([
        "First session message.",
        "/clear",
        "Second session message.",
    ]):
        await repl(session_id=None)

    all_sessions = Session.list_all()
    assert len(all_sessions) >= 2, (
        f"Expected at least 2 sessions after /clear, found: {len(all_sessions)}"
    )
    ids = [s["id"] for s in all_sessions]
    assert len(set(ids)) == len(ids), "Session IDs must be unique"


# ── /sessions command ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cli_sessions_command_lists_known_sessions(tmp_sessions, capsys):
    """/sessions must print the ID prefix of at least one known session."""
    # Pre-create a session to list
    pre = Session()
    pre.append(Turn.now("user", "seed"))

    with _patch_prompt(["/sessions"]):
        await repl(session_id=None)

    out = capsys.readouterr().out
    assert pre.session_id[:8] in out, (
        f"Pre-existing session {pre.session_id[:8]} not found in output:\n{out}"
    )


# ── drain on exit ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cli_drain_called_on_exit(tmp_sessions):
    """drain_pending_extraction must be called when the REPL exits."""
    drained: list[bool] = []

    async def _mock_drain(timeout_s: float = 30.0) -> None:
        drained.append(True)

    with _patch_prompt(["What is 2 + 2?"]):
        with patch("src.cli.drain_pending_extraction", side_effect=_mock_drain):
            await repl(session_id=None)

    assert drained, "drain_pending_extraction was not called on REPL exit"


# ── two independent CLI runs ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cli_two_runs_independent_sessions(tmp_sessions):
    """
    Two separate repl() calls must create two distinct sessions, each
    containing only their own turns.
    """
    # Run A
    with _patch_prompt(["I am a data engineer."]):
        await repl(session_id=None)
    sid_a = _latest_session().session_id
    init_extractor()

    # Run B
    with _patch_prompt(["I am a machine learning researcher."]):
        await repl(session_id=None)
    sid_b = _latest_session().session_id

    assert sid_a != sid_b

    ra = Session(sid_a).load()
    rb = Session(sid_b).load()

    assert "data engineer" in ra.history[0].content
    assert "machine learning" in rb.history[0].content
    # Neither session bleeds into the other
    assert len(ra.history) == 2
    assert len(rb.history) == 2
