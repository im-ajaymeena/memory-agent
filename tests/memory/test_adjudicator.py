import pathlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.adjudicator import adjudicate
from src.memory.models import Category, CandidateFact, MemoryRecord, Source
from src.memory.store import MemoryStore

import numpy as np


@pytest.fixture
def store(tmp_path: pathlib.Path) -> MemoryStore:
    return MemoryStore(tmp_path / "adj_test.db")


def _candidate(text: str = "User prefers Python", category: Category = Category.PREFERENCES) -> CandidateFact:
    return CandidateFact(
        text=text,
        category=category,
        source=Source.USER_STATEMENT,
        intent_label="tech_preference",
    )


def _mock_llm(operation: str, target_id: str | None = None) -> MagicMock:
    import json
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps({"operation": operation, "target_id": target_id}))]
    return msg


@pytest.mark.asyncio
async def test_add_inserts_new_record(store: MemoryStore) -> None:
    with patch("src.memory.adjudicator._client") as mock_client, \
         patch("src.memory.adjudicator.embed") as mock_embed:
        mock_embed.return_value = np.array([1.0, 0.0], dtype="float32")
        mock_client.messages.create = AsyncMock(return_value=_mock_llm("ADD"))
        await adjudicate(_candidate(), store)
    assert store.count_active() == 1


@pytest.mark.asyncio
async def test_none_does_not_write(store: MemoryStore) -> None:
    with patch("src.memory.adjudicator._client") as mock_client, \
         patch("src.memory.adjudicator.embed") as mock_embed:
        mock_embed.return_value = np.array([1.0, 0.0], dtype="float32")
        mock_client.messages.create = AsyncMock(return_value=_mock_llm("NONE"))
        await adjudicate(_candidate(), store)
    assert store.count_active() == 0


@pytest.mark.asyncio
async def test_update_changes_existing_record(store: MemoryStore, tmp_path: pathlib.Path) -> None:
    existing = MemoryRecord(
        text="User likes Python",
        category=Category.PREFERENCES,
        source=Source.USER_STATEMENT,
        embedding=[1.0, 0.0],
    )
    store.add(existing)

    with patch("src.memory.adjudicator._client") as mock_client, \
         patch("src.memory.adjudicator.embed") as mock_embed:
        mock_embed.return_value = np.array([0.9, 0.1], dtype="float32")
        mock_client.messages.create = AsyncMock(
            return_value=_mock_llm("UPDATE", existing.id)
        )
        await adjudicate(_candidate("User strongly prefers Python, especially for data work"), store)

    updated = store.get_by_id(existing.id)
    assert updated is not None
    assert "strongly" in updated.text


@pytest.mark.asyncio
async def test_delete_soft_deletes_and_adds_new(store: MemoryStore) -> None:
    old = MemoryRecord(
        text="User lives in NYC",
        category=Category.PERSONAL,
        source=Source.USER_STATEMENT,
        embedding=[1.0, 0.0],
    )
    store.add(old)

    with patch("src.memory.adjudicator._client") as mock_client, \
         patch("src.memory.adjudicator.embed") as mock_embed:
        mock_embed.return_value = np.array([0.8, 0.2], dtype="float32")
        mock_client.messages.create = AsyncMock(
            return_value=_mock_llm("DELETE", old.id)
        )
        await adjudicate(_candidate("User moved from NYC to Berlin", Category.PERSONAL), store)

    old_record = store.get_by_id(old.id)
    assert old_record is not None
    assert old_record.is_current is False
    assert store.count_active() == 1  # the new Berlin record


@pytest.mark.asyncio
async def test_source_trust_fast_path_no_llm_call(store: MemoryStore) -> None:
    """Higher-trust near-identical record → NONE without any LLM call."""
    high_trust = MemoryRecord(
        text="User prefers Python",
        category=Category.PREFERENCES,
        source=Source.USER_STATEMENT,  # trust=3
        embedding=[1.0, 0.0],
    )
    store.add(high_trust)

    low_trust_candidate = CandidateFact(
        text="User prefers Python",
        category=Category.PREFERENCES,
        source=Source.AGENT_INFERENCE,  # trust=2
    )

    with patch("src.memory.adjudicator._client") as mock_client, \
         patch("src.memory.adjudicator.embed") as mock_embed:
        mock_embed.return_value = np.array([1.0, 0.0], dtype="float32")
        await adjudicate(low_trust_candidate, store)
        mock_client.messages.create.assert_not_called()

    assert store.count_active() == 1  # no new record added
