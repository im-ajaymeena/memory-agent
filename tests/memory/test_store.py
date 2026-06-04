import pytest
import pathlib

from src.memory.models import Category, MemoryRecord, Source
from src.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: pathlib.Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


def _record(text: str = "User prefers Python", **kwargs) -> MemoryRecord:
    return MemoryRecord(
        text=text,
        category=kwargs.get("category", Category.PREFERENCES),
        source=kwargs.get("source", Source.USER_STATEMENT),
        embedding=[0.1, 0.2, 0.3],
    )


def test_add_and_count(store: MemoryStore) -> None:
    assert store.count_active() == 0
    store.add(_record())
    assert store.count_active() == 1


def test_soft_delete_excluded_from_search(store: MemoryStore) -> None:
    import numpy as np
    r = _record(embedding=[1.0, 0.0, 0.0])
    store.add(r)
    store.soft_delete(r.id)
    results = store.search(np.array([1.0, 0.0, 0.0], dtype="float32"), k=5)
    assert results == []
    assert store.count_active() == 0


def test_soft_delete_preserves_record_on_disk(store: MemoryStore) -> None:
    r = _record()
    store.add(r)
    store.soft_delete(r.id)
    # get_by_id fetches regardless of is_current
    fetched = store.get_by_id(r.id)
    assert fetched is not None
    assert fetched.is_current is False


def test_update_changes_text_and_timestamp(store: MemoryStore) -> None:
    import time
    r = _record()
    store.add(r)
    old_ts = r.timestamp_updated
    time.sleep(0.01)
    store.update(r.id, "User prefers TypeScript", [0.4, 0.5, 0.6])
    updated = store.get_by_id(r.id)
    assert updated is not None
    assert updated.text == "User prefers TypeScript"
    assert updated.timestamp_updated > old_ts


def test_persist_and_reload(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "mem.db"
    s1 = MemoryStore(db)
    s1.add(_record("User lives in Berlin"))
    del s1
    s2 = MemoryStore(db)
    results = s2.all_active()
    assert len(results) == 1
    assert results[0].text == "User lives in Berlin"


def test_search_returns_most_similar(store: MemoryStore) -> None:
    import numpy as np
    r1 = MemoryRecord(
        text="Python fan", category=Category.PREFERENCES,
        source=Source.USER_STATEMENT, embedding=[1.0, 0.0]
    )
    r2 = MemoryRecord(
        text="Lives in NYC", category=Category.PERSONAL,
        source=Source.USER_STATEMENT, embedding=[0.0, 1.0]
    )
    store.add(r1)
    store.add(r2)
    results = store.search(np.array([1.0, 0.0], dtype="float32"), k=1)
    assert results[0].id == r1.id


def test_category_filter(store: MemoryStore) -> None:
    import numpy as np
    store.add(MemoryRecord(
        text="Prefers Go", category=Category.PREFERENCES,
        source=Source.USER_STATEMENT, embedding=[1.0, 0.0]
    ))
    store.add(MemoryRecord(
        text="Works at Acme", category=Category.PROFESSIONAL,
        source=Source.USER_STATEMENT, embedding=[0.9, 0.1]
    ))
    results = store.search(
        np.array([1.0, 0.0], dtype="float32"), k=5,
        category=Category.PREFERENCES
    )
    assert all(r.category == Category.PREFERENCES for r in results)
    assert len(results) == 1
