"""
Integration tests: verify that schedule_extraction is called after chat()
and that turns are appended to the session before extraction runs.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.agent as agent_module
import src.extractor as extractor_module
from src.agent import chat
from src.extractor import drain_pending_extraction
from src.memory.stub import VanillaMemory
from src.session import Session


@pytest.fixture
def patched_stream(tmp_sessions):
    async def _fake_text_stream():
        yield "response"

    fake = MagicMock()
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=False)
    fake.text_stream = _fake_text_stream()

    with patch.object(agent_module.client.messages, "stream", return_value=fake):
        yield


@pytest.mark.asyncio
async def test_extract_called_after_chat(tmp_sessions, patched_stream):
    extracted: list[list] = []

    class CapturingMemory(VanillaMemory):
        async def extract_and_store(self, turns: list) -> None:
            extracted.append(list(turns))

    original = agent_module.memory
    agent_module.memory = CapturingMemory()
    extractor_module.set_memory(agent_module.memory)

    s = Session()
    await chat("hello", s)
    await drain_pending_extraction(timeout_s=5.0)

    agent_module.memory = original
    extractor_module.set_memory(original)

    assert len(extracted) == 1
    assert len(extracted[0]) == 2  # user + assistant turns


@pytest.mark.asyncio
async def test_turns_appended_to_session_after_chat(tmp_sessions, patched_stream):
    s = Session()
    assert len(s.history) == 0

    await chat("hello", s)
    await drain_pending_extraction(timeout_s=5.0)

    assert len(s.history) == 2
    assert s.history[0].role == "user"
    assert s.history[0].content == "hello"
    assert s.history[1].role == "assistant"
