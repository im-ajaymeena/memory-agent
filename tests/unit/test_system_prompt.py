from src.agent import _build_system_prompt
from src.memory.stub import Memory


def _mem(body: str, type_: str = "preferences_interests") -> Memory:
    return Memory(
        id="test-id",
        body=body,
        type=type_,
        source="user_statement",
        updated_at="2026-06-05T00:00:00+00:00",
        age_human_readable="1 day ago",
    )


def test_no_memories_returns_base():
    prompt = _build_system_prompt([])
    assert "helpful assistant" in prompt
    assert "Known facts" not in prompt


def test_memories_injected():
    memories = [_mem("User prefers Python."), _mem("User is based in London.")]
    prompt = _build_system_prompt(memories)
    assert "User prefers Python." in prompt
    assert "User is based in London." in prompt
    assert "Known facts about this user:" in prompt


def test_memory_format_includes_type_and_age():
    m = _mem("User likes dark mode.", "preferences_interests")
    prompt = _build_system_prompt([m])
    assert "[preferences_interests | 1 day ago]" in prompt


def test_multiple_memories_all_present():
    memories = [_mem(f"fact {i}", "test_type") for i in range(5)]
    prompt = _build_system_prompt(memories)
    for i in range(5):
        assert f"fact {i}" in prompt
