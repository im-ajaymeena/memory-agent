"""
Integration tests: verify memory.retrieve() is called with the user's query
and that its results reach the system prompt.
No real LLM calls — Anthropic client is mocked at the boundary.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.agent as agent_module
from src.agent import _build_system_prompt, chat
from src.memory.stub import Memory, VanillaMemory
from src.session import Session


def _make_memory(body: str) -> Memory:
    return Memory(
        id="m1",
        body=body,
        type="preferences_interests",
        source="user_statement",
        updated_at="2026-06-05T00:00:00+00:00",
        age_human_readable="1 day ago",
    )


@pytest.fixture
def patched_stream(tmp_sessions):
    """Replace AsyncAnthropic streaming with a stub that yields one token."""
    async def _fake_text_stream():
        yield "hello"

    fake_stream = MagicMock()
    fake_stream.__aenter__ = AsyncMock(return_value=fake_stream)
    fake_stream.__aexit__ = AsyncMock(return_value=False)
    fake_stream.text_stream = _fake_text_stream()

    with patch.object(agent_module.client.messages, "stream", return_value=fake_stream):
        yield


@pytest.mark.asyncio
async def test_retrieve_called_with_user_query(tmp_sessions, patched_stream):
    queries_seen: list[str] = []

    class TrackingMemory(VanillaMemory):
        def retrieve(self, query: str) -> list[Memory]:
            queries_seen.append(query)
            return []

    original = agent_module.memory
    agent_module.memory = TrackingMemory()

    s = Session()
    await chat("what language do I prefer?", s)

    agent_module.memory = original
    assert "what language do I prefer?" in queries_seen


@pytest.mark.asyncio
async def test_retrieved_memories_appear_in_system_prompt(tmp_sessions, patched_stream):
    mem = _make_memory("User prefers Python for all scripting.")

    captured_system: list[str] = []

    class InjectingMemory(VanillaMemory):
        def retrieve(self, query: str) -> list[Memory]:
            return [mem]

    def capturing_stream(**kwargs):
        captured_system.append(kwargs.get("system", ""))
        async def _text():
            yield "ok"
        fake = MagicMock()
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=False)
        fake.text_stream = _text()
        return fake

    original = agent_module.memory
    agent_module.memory = InjectingMemory()

    with patch.object(agent_module.client.messages, "stream", side_effect=capturing_stream):
        s = Session()
        await chat("what do I prefer?", s)

    agent_module.memory = original
    assert any("User prefers Python for all scripting." in sys for sys in captured_system)
