"""
E2E test: memory established in session 1 shapes agent behavior in session 2.

Requires a real ANTHROPIC_API_KEY. Marked @pytest.mark.slow — excluded from
the default test run (make test). Run with: make test-e2e
"""
import asyncio

import pytest

from src.agent import _build_system_prompt, chat
from src.memory.stub import Memory, VanillaMemory
from src.session import Session


pytestmark = pytest.mark.slow


def _inject_memory(mem: VanillaMemory, body: str, type_: str = "preferences_interests") -> None:
    """Directly inject a memory (bypasses extraction — stub only)."""
    mem._store.append(
        Memory(
            id="e2e-test",
            body=body,
            type=type_,
            source="user_statement",
            updated_at="2026-06-05T00:00:00+00:00",
            age_human_readable="just now",
        )
    )


@pytest.mark.asyncio
async def test_memory_shapes_second_session_response(tmp_sessions):
    """
    Inject a user preference into the memory store, then start a new session
    and verify the agent references that preference in its response.
    """
    import src.agent as agent_module

    # agent_module.memory is RealMemory; use an explicit VanillaMemory stub so
    # _inject_memory can use its list-backed _store without touching the real DB.
    stub = VanillaMemory()
    original = agent_module.memory
    agent_module.memory = stub
    try:
        _inject_memory(stub, "User strongly prefers Python over all other languages.")

        s2 = Session()
        response, _ = await chat("What language should I use for my next script?", s2)

        assert "python" in response.lower(), (
            f"Expected 'python' in response based on injected memory, got:\n{response}"
        )
    finally:
        agent_module.memory = original
