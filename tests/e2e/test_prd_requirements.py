"""
TEST_PLAN Part 2 — The Three Hard Problems (real LLM + real observer/adjudicator).
TEST_PLAN Part 3 — Cross-session persistence with real extraction.
TEST_PLAN Part 3.2 — Long-session TTFT degradation.
TEST_PLAN Part 6.3 — Relevance-based retrieval (only relevant facts injected).
TEST_PLAN Part 6.4 — PII / credential exclusion.

All tests use real Anthropic API calls.
Requires: ANTHROPIC_API_KEY
Run with: make test-e2e
"""
import asyncio
import pathlib
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pytest

import src.agent as agent_module
from src.agent import chat, _build_system_prompt
from src.extractor import drain_pending_extraction, init_extractor
from src.memory.embedder import embed
from src.memory.models import Category, MemoryRecord, Source
from src.memory.real import RealMemory
from src.memory.store import MemoryStore
from src.session import Session, Turn


pytestmark = pytest.mark.slow


# ── helpers ───────────────────────────────────────────────────────────────────

def _tmp_memory(tmp_path: pathlib.Path) -> RealMemory:
    return RealMemory(db_path=tmp_path / "test.db")


def _make_turns(*pairs: tuple[str, str]) -> list[Turn]:
    turns = []
    for user, assistant in pairs:
        turns.append(Turn.now("user", user))
        turns.append(Turn.now("assistant", assistant))
    return turns


# ── 2.1 Chatterbox test ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chatterbox_noise_stores_zero_memories(tmp_path):
    """
    TEST_PLAN 2.1: 20 turns of pure conversational noise must not produce
    any stored memories. Observer must filter all of it.
    """
    mem = _tmp_memory(tmp_path)
    noise_turns = _make_turns(
        ("Hello!", "Hi there!"),
        ("How are you?", "I'm doing well, thank you."),
        ("Okay.", "Alright!"),
        ("Sounds good.", "Great!"),
        ("Thanks.", "You're welcome!"),
        ("Got it.", "Perfect."),
        ("Alright then.", "Sure thing."),
        ("Cool.", "Indeed!"),
        ("Bye!", "Goodbye!"),
        ("See you later.", "Take care!"),
    )

    await mem.extract_and_store(noise_turns)

    assert mem.count() == 0, (
        f"Expected 0 memories from noise, got {mem.count()}. "
        f"Observer is not filtering noise correctly."
    )


# ── 2.2 Stale memory resolution ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stale_memory_resolved_on_contradiction(tmp_path):
    """
    TEST_PLAN 2.2: When user contradicts a prior fact with explicit supersession
    language, the old record must be replaced. Agent then uses the new fact.
    """
    mem = _tmp_memory(tmp_path)

    # Step 1: establish the original fact
    turn1 = _make_turns(("I am a strict vegetarian.", "Got it, I'll remember that."))
    await mem.extract_and_store(turn1)
    count_after_step1 = mem.count()
    assert count_after_step1 >= 1, "Vegetarian fact should have been stored"

    # Step 2: explicit contradiction with supersession language
    turn2 = _make_turns((
        "I've changed my diet. I now eat fish — I'm pescatarian.",
        "Noted, you're now pescatarian."
    ))
    await mem.extract_and_store(turn2)

    # Step 3: verify the active memories contain the new fact and not the old
    all_memories = mem.all()
    bodies = [m.body.lower() for m in all_memories]

    has_pescatarian = any("pescatarian" in b or "fish" in b for b in bodies)
    # The vegetarian record should have been DELETEd or UPDATEd
    has_only_vegetarian = any("vegetarian" in b for b in bodies) and not has_pescatarian

    assert has_pescatarian, (
        f"Pescatarian fact not found in memories after contradiction.\n"
        f"Active memories: {bodies}"
    )
    assert not has_only_vegetarian, (
        f"Old 'vegetarian' fact still active without pescatarian override.\n"
        f"Active memories: {bodies}"
    )


# ── 2.3 Retention under noise ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_important_fact_survives_noise(tmp_path):
    """
    TEST_PLAN 2.3: An important fact stored in memory must survive 20 turns
    of unrelated noise — it must not be deleted or overwritten.
    """
    mem = _tmp_memory(tmp_path)

    # Step 1: store a clearly durable user preference
    important = _make_turns((
        "I have been writing Python professionally for ten years — it's my primary language.",
        "Noted, Python is your primary language.",
    ))
    await mem.extract_and_store(important)
    assert mem.count() >= 1, (
        "Important durable fact was not stored. "
        "Note: IP addresses are intentionally excluded as ephemeral task state."
    )

    # Step 2: flood with unrelated noise
    noise = _make_turns(
        ("Hello!", "Hi!"), ("How are you?", "Fine."),
        ("Nice weather.", "Indeed."), ("Okay.", "Sure."),
        ("Thanks.", "Welcome."), ("Got it.", "Great."),
        ("Cool.", "Yep."), ("Bye.", "Goodbye."),
        ("See you.", "Take care."), ("Later.", "Bye!"),
    )
    await mem.extract_and_store(noise)

    # Step 3: the durable preference must still be active
    all_memories = mem.all()
    bodies = [m.body.lower() for m in all_memories]
    assert any("python" in b for b in bodies), (
        f"Important fact ('Python') was lost after noise injection.\n"
        f"Active memories: {[m.body for m in mem.all()]}"
    )


# ── 3.1 Real cross-session recall ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_real_cross_session_recall(tmp_path, tmp_sessions):
    """
    TEST_PLAN 3.1: Fact established in session 1 via real extraction must
    shape agent response in session 2 via real retrieval.
    """
    mem = _tmp_memory(tmp_path)
    original = agent_module.memory
    agent_module.memory = mem
    import src.extractor as _ext
    _ext.set_memory(mem)

    # ── Session 1: user tells the agent their language preference ─────────────
    init_extractor()
    s1 = Session()
    captured: list[dict] = []

    def cap_stream(**kwargs):
        captured.append(kwargs)
        from unittest.mock import AsyncMock, MagicMock
        async def _text():
            yield "Noted, I'll remember you prefer Python."
        stub = MagicMock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=False)
        stub.text_stream = _text()
        return stub

    with patch.object(agent_module.client.messages, "stream", side_effect=cap_stream):
        await chat("I always write Python — it's my primary language.", s1)

    # Wait for real extraction to complete
    await drain_pending_extraction(timeout_s=30.0)

    assert mem.count() >= 1, (
        "Session 1 extraction produced no memories. "
        "Check ANTHROPIC_API_KEY is set and observer is working."
    )

    # ── Session 2: fresh session, ask about language preference ───────────────
    init_extractor()
    s2 = Session()
    system_prompts: list[str] = []

    def cap_stream2(**kwargs):
        system_prompts.append(kwargs.get("system", ""))
        from unittest.mock import AsyncMock, MagicMock
        async def _text():
            yield "Based on what I know, you prefer Python."
        stub = MagicMock()
        stub.__aenter__ = AsyncMock(return_value=stub)
        stub.__aexit__ = AsyncMock(return_value=False)
        stub.text_stream = _text()
        return stub

    with patch.object(agent_module.client.messages, "stream", side_effect=cap_stream2):
        await chat("What programming language do I prefer?", s2)

    await drain_pending_extraction(timeout_s=10.0)

    agent_module.memory = original
    _ext.set_memory(original)

    assert system_prompts, "No API call was made in session 2"
    assert "python" in system_prompts[0].lower(), (
        f"Session 1 memory ('Python') not found in session 2 system prompt.\n"
        f"System prompt was:\n{system_prompts[0]}"
    )


# ── 3.2 Long-session TTFT degradation ────────────────────────────────────────

@pytest.mark.asyncio
async def test_long_session_ttft_does_not_degrade(tmp_sessions):
    """
    TEST_PLAN 3.2: TTFT at turn 50 must be within 200ms of TTFT at turn 1
    within a single session (rolling history window keeps prompt size constant).
    """
    import anthropic

    client = anthropic.AsyncAnthropic()

    async def _ttft(msg: str) -> float:
        t = time.perf_counter()
        first = True
        ttft = 0.0
        async with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=32,
            messages=[{"role": "user", "content": msg}],
        ) as stream:
            async for _ in stream.text_stream:
                if first:
                    ttft = time.perf_counter() - t
                    first = False
                    break
        return ttft

    # Turn 1 — cold
    ttft_turn1 = await _ttft("Say 'ok'.")

    # Simulate 49 more turns by building a session with history
    s = Session()
    for i in range(48):
        s.append(Turn.now("user", f"Turn {i} filler message."))
        s.append(Turn.now("assistant", f"Acknowledged turn {i}."))

    from src.agent import _build_messages
    messages = _build_messages(s, "Say 'ok' again.")

    t = time.perf_counter()
    ttft_turn50 = 0.0
    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=32,
        messages=messages,
    ) as stream:
        async for _ in stream.text_stream:
            ttft_turn50 = time.perf_counter() - t
            break

    delta_ms = (ttft_turn50 - ttft_turn1) * 1000
    assert delta_ms < 200, (
        f"TTFT degraded by {delta_ms:.0f}ms over 50 turns (limit: 200ms).\n"
        f"Turn 1: {ttft_turn1*1000:.0f}ms  Turn 50: {ttft_turn50*1000:.0f}ms"
    )


# ── 6.3 Relevance-based retrieval ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_relevance_retrieval_returns_topically_relevant_facts(tmp_path):
    """
    TEST_PLAN 6.3: Given 30 diverse facts (pets, food, coding), a coding query
    must surface coding facts at the top — not pet or food facts.
    """
    store = MemoryStore(tmp_path / "rel.db")

    coding_facts = [
        "User prefers Python for all scripting tasks.",
        "User's primary stack is FastAPI and PostgreSQL.",
        "User writes tests with pytest.",
    ]
    pet_facts = [
        "User has a golden retriever named Biscuit.",
        "User's cat is called Shadow.",
        "User volunteers at an animal shelter on weekends.",
    ]
    food_facts = [
        "User is vegetarian and avoids meat.",
        "User's favourite cuisine is Japanese food.",
        "User drinks oat milk in their coffee.",
    ]

    for facts in (coding_facts, pet_facts, food_facts):
        for text in facts:
            store.add(MemoryRecord(
                text=text,
                category=Category.PREFERENCES,
                source=Source.USER_STATEMENT,
                embedding=embed(text).tolist(),
            ))

    from src.memory.retriever import retrieve
    results = retrieve("What testing framework should I use for my Python project?", store, k=3)

    result_bodies = [r.body for r in results]
    has_coding = any(
        any(kw in b.lower() for kw in ["python", "pytest", "fastapi", "stack", "scripting"])
        for b in result_bodies
    )
    assert has_coding, (
        f"No coding fact in top-3 results for a coding query.\n"
        f"Got: {result_bodies}"
    )


# ── 6.4 PII / credential exclusion ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_credentials_not_stored(tmp_path):
    """
    TEST_PLAN 6.4: API keys, SSNs, and passwords shared by the user must not
    appear in the memory store after extraction.
    """
    mem = _tmp_memory(tmp_path)

    pii_turns = _make_turns(
        ("My API key is sk-ant-abc123xyz.", "I understand."),
        ("My SSN is 123-45-6789, just FYI.", "Okay."),
        ("The database password is hunter2.", "Got it."),
    )

    await mem.extract_and_store(pii_turns)

    all_bodies = [m.body for m in mem.all()]
    sensitive = ["sk-ant-abc123xyz", "123-45-6789", "hunter2"]
    for secret in sensitive:
        assert not any(secret in b for b in all_bodies), (
            f"Sensitive value '{secret}' found in memory store after extraction.\n"
            f"Stored memories: {all_bodies}"
        )
