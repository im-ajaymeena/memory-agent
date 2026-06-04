"""
Behavioural I/O tests — full CLI + agent pipeline with mocked LLM boundary.

These tests verify *what goes into* the API (messages shape, system prompt
content, history ordering) and *what comes out* of the system (JSONL on disk,
session state, extractor calls) without caring about LLM response content.

No real API calls → deterministic, fast, no API key required.
"""
import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

import src.agent as agent_module
import src.extractor as extractor_module
from src.agent import _build_messages, _build_system_prompt, chat
from src.cli import repl
from src.extractor import drain_pending_extraction
from src.memory.stub import Memory, VanillaMemory
from src.session import Session, Turn


# ── LLM stub ─────────────────────────────────────────────────────────────────

def _make_stream_stub(response_text: str = "stub response"):
    """Returns a mock that behaves like client.messages.stream(...)."""
    async def _text_stream():
        yield response_text

    stub = MagicMock()
    stub.__aenter__ = AsyncMock(return_value=stub)
    stub.__aexit__ = AsyncMock(return_value=False)
    stub.text_stream = _text_stream()
    return stub


def _patch_stream(response_text: str = "stub response"):
    stub = _make_stream_stub(response_text)
    return patch.object(agent_module.client.messages, "stream", return_value=stub)


def _fake_prompt(inputs: list[str]):
    q = deque(inputs)

    async def _prompt_async(*args, **kwargs):
        if q:
            return q.popleft()
        raise EOFError

    return _prompt_async


def _patch_prompt(inputs: list[str]):
    return patch("src.cli.PromptSession.prompt_async", new=_fake_prompt(inputs))


# ── messages shape sent to the API ───────────────────────────────────────────

def test_messages_role_alternation(tmp_sessions):
    """
    _build_messages must produce strictly alternating user/assistant roles
    followed by the new user message.
    """
    s = Session()
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        s.append(Turn.now(role, f"turn {i}"))

    msgs = _build_messages(s, "new input")

    roles = [m["role"] for m in msgs]
    assert roles[-1] == "user"
    assert roles[-2] == "assistant"
    # All roles must be valid
    assert all(r in {"user", "assistant"} for r in roles)


def test_messages_newest_turn_is_last_history_entry(tmp_sessions):
    s = Session()
    s.append(Turn.now("user", "first"))
    s.append(Turn.now("assistant", "second"))
    s.append(Turn.now("user", "third"))

    msgs = _build_messages(s, "fourth")

    assert msgs[-2]["content"] == "third"
    assert msgs[-1]["content"] == "fourth"


def test_messages_new_input_always_appended_as_user(tmp_sessions):
    s = Session()
    msgs = _build_messages(s, "hello")
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_messages_content_preserved_exactly(tmp_sessions):
    """Content must be passed verbatim — no trimming or modification."""
    s = Session()
    text = "  leading spaces and unicode 🎉  "
    s.append(Turn.now("user", text))

    msgs = _build_messages(s, "q")
    assert msgs[0]["content"] == text


# ── system prompt structure ───────────────────────────────────────────────────

def test_system_prompt_no_memories_is_base_only():
    prompt = _build_system_prompt([])
    assert "Known facts" not in prompt


def test_system_prompt_memory_format():
    m = Memory(
        id="x", body="User uses vim.", type="tools",
        source="user_statement", updated_at="2026-06-05T00:00:00+00:00",
        age_human_readable="5 mins ago",
    )
    prompt = _build_system_prompt([m])
    assert "[tools | 5 mins ago] User uses vim." in prompt


def test_system_prompt_all_memories_present():
    mems = [
        Memory(id=str(i), body=f"fact {i}", type="t", source="s",
               updated_at="2026-06-05T00:00:00+00:00", age_human_readable="now")
        for i in range(5)
    ]
    prompt = _build_system_prompt(mems)
    for i in range(5):
        assert f"fact {i}" in prompt


# ── what reaches the API ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_api_receives_correct_messages(tmp_sessions):
    """Verify the exact messages list sent to the API on a two-turn session."""
    captured: list[dict] = []

    def capturing_stream(**kwargs):
        captured.append(kwargs)
        return _make_stream_stub("reply")

    s = Session()
    # Pre-load one prior turn (simulates resumed session)
    s.append(Turn.now("user", "prior user turn"))
    s.append(Turn.now("assistant", "prior assistant turn"))

    with patch.object(agent_module.client.messages, "stream", side_effect=capturing_stream):
        await chat("new question", s)
    await drain_pending_extraction(timeout_s=5.0)

    assert captured, "stream() was never called"
    msgs = captured[0]["messages"]

    assert msgs[0] == {"role": "user", "content": "prior user turn"}
    assert msgs[1] == {"role": "assistant", "content": "prior assistant turn"}
    assert msgs[2] == {"role": "user", "content": "new question"}


@pytest.mark.asyncio
async def test_api_receives_system_prompt(tmp_sessions):
    captured: list[dict] = []

    def capturing_stream(**kwargs):
        captured.append(kwargs)
        return _make_stream_stub("ok")

    s = Session()
    with patch.object(agent_module.client.messages, "stream", side_effect=capturing_stream):
        await chat("hello", s)
    await drain_pending_extraction(timeout_s=5.0)

    assert "system" in captured[0]
    assert len(captured[0]["system"]) > 0


@pytest.mark.asyncio
async def test_api_receives_memories_in_system_prompt(tmp_sessions):
    """When memory.retrieve returns facts, they appear in the system prompt sent to API."""
    captured: list[dict] = []

    def capturing_stream(**kwargs):
        captured.append(kwargs)
        return _make_stream_stub("ok")

    injected = Memory(
        id="m1", body="User is an SRE.", type="professional_details",
        source="user_statement", updated_at="2026-06-05T00:00:00+00:00",
        age_human_readable="2 hours ago",
    )

    class _InjectedMemory(VanillaMemory):
        def retrieve(self, query: str) -> list[Memory]:
            return [injected]

    original = agent_module.memory
    agent_module.memory = _InjectedMemory()

    s = Session()
    with patch.object(agent_module.client.messages, "stream", side_effect=capturing_stream):
        await chat("what do you know about me?", s)
    await drain_pending_extraction(timeout_s=5.0)

    agent_module.memory = original

    system = captured[0]["system"]
    assert "User is an SRE." in system
    assert "professional_details" in system


# ── session JSONL on disk ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jsonl_contains_all_fields(tmp_sessions):
    """Every persisted turn must have role, content, timestamp, id."""
    import json

    s = Session()
    with _patch_stream("test reply"):
        await chat("test input", s)
    await drain_pending_extraction(timeout_s=5.0)

    lines = [json.loads(l) for l in s.path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2

    for line in lines:
        assert "role" in line
        assert "content" in line
        assert "timestamp" in line
        assert "id" in line
        assert line["id"] != ""


@pytest.mark.asyncio
async def test_jsonl_order_is_user_then_assistant(tmp_sessions):
    import json

    s = Session()
    with _patch_stream("assistant reply"):
        await chat("user input", s)
    await drain_pending_extraction(timeout_s=5.0)

    lines = [json.loads(l) for l in s.path.read_text().splitlines() if l.strip()]
    assert lines[0]["role"] == "user"
    assert lines[0]["content"] == "user input"
    assert lines[1]["role"] == "assistant"
    assert lines[1]["content"] == "assistant reply"


# ── CLI pipeline I/O ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cli_pipeline_multi_turn_messages_shape(tmp_sessions):
    """
    Full CLI pipeline with mocked LLM: verify that turn 2's API call includes
    turn 1's user+assistant pair in the messages list.
    """
    call_count = 0
    messages_per_call: list[list] = []

    def capturing_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        messages_per_call.append(kwargs["messages"])

        async def _text():
            yield f"reply {call_count}"

        stub = MagicMock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=False)
        stub.text_stream = _text()
        return stub

    with _patch_prompt(["first message", "second message"]):
        with patch.object(agent_module.client.messages, "stream", side_effect=capturing_stream):
            await repl(session_id=None)

    assert call_count == 2

    # Turn 1: only the user message
    assert messages_per_call[0][-1]["content"] == "first message"
    assert len(messages_per_call[0]) == 1

    # Turn 2: prior user + assistant + new user
    assert len(messages_per_call[1]) == 3
    assert messages_per_call[1][0]["role"] == "user"
    assert messages_per_call[1][0]["content"] == "first message"
    assert messages_per_call[1][1]["role"] == "assistant"
    assert messages_per_call[1][1]["content"] == "reply 1"
    assert messages_per_call[1][2]["role"] == "user"
    assert messages_per_call[1][2]["content"] == "second message"


@pytest.mark.asyncio
async def test_cli_pipeline_session_resume_sends_prior_history(tmp_sessions):
    """
    Resume an existing session: the API call must include the prior turns
    from the loaded session in the messages list.
    """
    s = Session()
    s.append(Turn.now("user", "I work in fintech."))
    s.append(Turn.now("assistant", "Noted, you work in fintech."))

    messages_received: list[list] = []

    def capturing_stream(**kwargs):
        messages_received.append(kwargs["messages"])
        return _make_stream_stub("ok")

    with _patch_prompt(["What sector do I work in?"]):
        with patch.object(agent_module.client.messages, "stream", side_effect=capturing_stream):
            await repl(session_id=s.session_id)

    assert messages_received
    msgs = messages_received[0]

    contents = [m["content"] for m in msgs]
    assert "I work in fintech." in contents
    assert "Noted, you work in fintech." in contents
    assert "What sector do I work in?" in contents


@pytest.mark.asyncio
async def test_cli_clear_resets_history_sent_to_api(tmp_sessions):
    """
    After /clear, the next API call must NOT include turns from the first session.
    """
    messages_per_call: list[list] = []
    call_count = 0

    def capturing_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        messages_per_call.append(kwargs["messages"])

        async def _text():
            yield f"reply {call_count}"

        stub = MagicMock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=False)
        stub.text_stream = _text()
        return stub

    with _patch_prompt(["first session message", "/clear", "second session message"]):
        with patch.object(agent_module.client.messages, "stream", side_effect=capturing_stream):
            await repl(session_id=None)

    assert call_count == 2

    # First call: only first session message
    assert messages_per_call[0][-1]["content"] == "first session message"

    # Second call (after /clear): must NOT contain first session's content
    second_contents = [m["content"] for m in messages_per_call[1]]
    assert "first session message" not in second_contents
    assert "second session message" in second_contents
