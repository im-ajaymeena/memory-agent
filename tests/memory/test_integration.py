"""
Integration tests for RealMemory.
All LLM calls mocked — no API key required.
"""
import json
import pathlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.real import RealMemory
from src.session import Turn


def _turn(role: str, content: str) -> Turn:
    return Turn.now(role, content)


def _observer_says(facts: list[dict]) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(facts))]
    return msg


def _adjudicator_says(op: str, target_id: str | None = None) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps({"operation": op, "target_id": target_id}))]
    return msg


@pytest.fixture
def memory(tmp_path: pathlib.Path) -> RealMemory:
    return RealMemory(db_path=tmp_path / "test.db")


@pytest.mark.asyncio
async def test_extract_and_store_adds_record(memory: RealMemory) -> None:
    turns = [
        _turn("user", "I prefer TypeScript over JavaScript."),
        _turn("assistant", "Got it, I'll use TypeScript in my examples."),
    ]
    observer_payload = [{
        "text": "User prefers TypeScript over JavaScript.",
        "category": "preferences_interests",
        "source": "user_statement",
        "intent_label": "tech_preference",
        "entities": ["TypeScript", "JavaScript"],
    }]
    with patch("src.memory.real.observe", new_callable=AsyncMock) as mock_obs, \
         patch("src.memory.real.adjudicate", new_callable=AsyncMock) as mock_adj, \
         patch("src.memory.observer._client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_observer_says(observer_payload))
        from src.memory.models import CandidateFact, Category, Source
        mock_obs.return_value = [CandidateFact(
            text="User prefers TypeScript over JavaScript.",
            category=Category.PREFERENCES,
            source=Source.USER_STATEMENT,
        )]
        await memory.extract_and_store(turns)
        mock_obs.assert_called_once()
        mock_adj.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_returns_stored_memories(memory: RealMemory) -> None:
    from src.memory.models import Category, MemoryRecord, Source
    record = MemoryRecord(
        text="User is a senior engineer at Acme Corp.",
        category=Category.PROFESSIONAL,
        source=Source.USER_STATEMENT,
        embedding=[0.1] * 384,  # dummy 1536-dim vector matching text-embedding-3-small
    )
    memory._store.add(record)

    with patch("src.memory.retriever.embed") as mock_embed:
        import numpy as np
        mock_embed.return_value = np.array([0.1] * 384, dtype="float32")
        results = memory.retrieve("What is the user's job?")

    assert len(results) == 1
    assert "Acme Corp" in results[0].body
    assert results[0].type == "professional_details"


@pytest.mark.asyncio
async def test_cross_session_persistence(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "cross_session.db"

    # Session 1: write a record directly
    mem1 = RealMemory(db_path=db)
    from src.memory.models import Category, MemoryRecord, Source
    mem1._store.add(MemoryRecord(
        text="User's name is Alice.",
        category=Category.PERSONAL,
        source=Source.USER_STATEMENT,
        embedding=[0.5] * 384,
    ))
    assert mem1.count() == 1
    del mem1

    # Session 2: load from same path — record must survive
    mem2 = RealMemory(db_path=db)
    assert mem2.count() == 1
    with patch("src.memory.retriever.embed") as mock_embed:
        import numpy as np
        mock_embed.return_value = np.array([0.5] * 384, dtype="float32")
        results = mem2.retrieve("What is my name?")
    assert any("Alice" in r.body for r in results)


@pytest.mark.asyncio
async def test_noise_turns_produce_no_writes(memory: RealMemory) -> None:
    turns = [
        _turn("user", "Thanks!"),
        _turn("assistant", "You're welcome!"),
    ]
    with patch("src.memory.real.observe", new_callable=AsyncMock) as mock_obs, \
         patch("src.memory.real.adjudicate", new_callable=AsyncMock) as mock_adj:
        mock_obs.return_value = []  # observer filters everything
        await memory.extract_and_store(turns)
        mock_adj.assert_not_called()

    assert memory.count() == 0


@pytest.mark.asyncio
async def test_deleted_record_not_retrieved(memory: RealMemory) -> None:
    import numpy as np
    from src.memory.models import Category, MemoryRecord, Source

    old = MemoryRecord(
        text="User lives in New York.",
        category=Category.PERSONAL,
        source=Source.USER_STATEMENT,
        embedding=[1.0] + [0.0] * 383,
    )
    memory._store.add(old)
    memory._store.soft_delete(old.id)

    with patch("src.memory.retriever.embed") as mock_embed:
        mock_embed.return_value = np.array([1.0] + [0.0] * 383, dtype="float32")
        results = memory.retrieve("Where does the user live?")

    assert results == []
