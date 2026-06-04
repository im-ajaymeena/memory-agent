"""
E2E tests — agent core, session persistence, multi-turn coherence.
All use real LLM calls with VanillaMemory (no memory dependency).
Requires: ANTHROPIC_API_KEY

Run with: make test-e2e
"""
import asyncio
import re

import pytest

import src.agent as agent_module
from src.agent import chat
from src.extractor import drain_pending_extraction, init_extractor
from src.memory.stub import VanillaMemory
from src.session import Session, Turn


pytestmark = pytest.mark.slow


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_memory():
    """Ensure VanillaMemory is a fresh empty store for every e2e test."""
    original = agent_module.memory
    agent_module.memory = VanillaMemory()
    yield
    agent_module.memory = original


# ── smoke ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_returns_non_empty_response(tmp_sessions):
    s = Session()
    response, ttft = await chat("Say exactly: hello world", s)
    await drain_pending_extraction(timeout_s=5.0)

    assert len(response) > 0
    assert ttft > 0.0


@pytest.mark.asyncio
async def test_ttft_is_measured_and_positive(tmp_sessions):
    s = Session()
    _, ttft = await chat("What is 2 + 2?", s)
    await drain_pending_extraction(timeout_s=5.0)

    assert ttft > 0.0
    assert ttft < 30.0  # sanity ceiling


# ── session persistence ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_turns_written_to_disk_after_chat(tmp_sessions):
    s = Session()
    await chat("My favourite colour is blue.", s)
    await drain_pending_extraction(timeout_s=5.0)

    reloaded = Session(s.session_id).load()
    assert len(reloaded.history) == 2
    assert reloaded.history[0].role == "user"
    assert reloaded.history[0].content == "My favourite colour is blue."
    assert reloaded.history[1].role == "assistant"
    assert len(reloaded.history[1].content) > 0


@pytest.mark.asyncio
async def test_session_survives_simulated_restart(tmp_sessions):
    """Write turns in one Session object, reload in a new one — history intact."""
    s1 = Session()
    await chat("Remember: my project is called Nighthawk.", s1)
    await drain_pending_extraction(timeout_s=5.0)

    # Simulated restart: new object, same ID
    s2 = Session(s1.session_id).load()
    assert len(s2.history) == 2
    assert "Nighthawk" in s2.history[0].content


@pytest.mark.asyncio
async def test_multiple_sessions_are_independent(tmp_sessions):
    sa = Session()
    sb = Session()

    await chat("I prefer tabs.", sa)
    await chat("I prefer spaces.", sb)
    await drain_pending_extraction(timeout_s=10.0)

    ra = Session(sa.session_id).load()
    rb = Session(sb.session_id).load()

    assert "tabs" in ra.history[0].content
    assert "spaces" in rb.history[0].content
    assert ra.session_id != rb.session_id


# ── multi-turn coherence ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_turn_context_maintained(tmp_sessions):
    """
    The agent should reference the animal named in turn 1 when asked in turn 2.
    Tests that history is correctly sent to the API on subsequent turns.
    """
    s = Session()
    await chat("My pet's name is Biscuit. She's a golden retriever.", s)
    response, _ = await chat("What's my pet's name?", s)
    await drain_pending_extraction(timeout_s=5.0)

    assert "biscuit" in response.lower(), (
        f"Expected agent to recall 'Biscuit' from prior turn, got: {response}"
    )


@pytest.mark.asyncio
async def test_turn_ids_are_unique_across_multi_turn(tmp_sessions):
    s = Session()
    await chat("First message.", s)
    await chat("Second message.", s)
    await drain_pending_extraction(timeout_s=5.0)

    ids = [t.id for t in s.history]
    assert len(ids) == len(set(ids)), "Turn IDs must be unique"
    assert all(id_ != "" for id_ in ids)


# ── abort ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_abort_stops_stream_gracefully(tmp_sessions):
    """
    Abort signal set mid-stream should produce a partial (possibly empty)
    response without raising an exception.
    """
    s = Session()
    abort = asyncio.Event()

    async def _set_abort_soon():
        await asyncio.sleep(0.05)
        abort.set()

    asyncio.create_task(_set_abort_soon())
    response, _ = await chat(
        "Count slowly from 1 to 100, one number per line.", s, abort=abort
    )
    await drain_pending_extraction(timeout_s=5.0)

    # Response may be empty or partial — both are correct; no exception is the key.
    assert isinstance(response, str)


# ── rolling history window ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rolling_window_does_not_exceed_max_turns(tmp_sessions):
    """
    After MAX_TURNS_VERBATIM + 5 turns the messages list still fits within
    the rolling window limit.
    """
    from src.agent import MAX_TURNS_VERBATIM, _build_messages

    s = Session()
    for i in range(MAX_TURNS_VERBATIM + 5):
        role = "user" if i % 2 == 0 else "assistant"
        s.append(Turn.now(role, f"turn {i}"))

    messages = _build_messages(s, "new question")
    # MAX_TURNS_VERBATIM history turns + 1 new user turn
    assert len(messages) == MAX_TURNS_VERBATIM + 1
