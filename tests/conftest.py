import pathlib
from unittest.mock import AsyncMock, MagicMock

# Load .env before any src module is imported — observer/adjudicator create
# anthropic.AsyncAnthropic() at module level and need the key at import time.
from dotenv import load_dotenv
load_dotenv()

import pytest

import src.extractor as extractor_module
from src.memory.stub import Memory, VanillaMemory
from src.session import Session


@pytest.fixture
def tmp_sessions(tmp_path, monkeypatch):
    """Redirect session storage to a temp dir so tests don't touch ~/.agent."""
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr("src.session.SESSIONS_DIR", sessions_dir)
    return sessions_dir


@pytest.fixture(autouse=True)
def reset_extractor():
    """
    Reset all extractor closure state before and after every test.
    Also injects a no-op memory stub so unit/integration tests never
    make real LLM calls during background extraction.
    """
    noop = VanillaMemory()
    extractor_module.set_memory(noop)
    extractor_module.init_extractor()
    yield
    extractor_module.init_extractor()


@pytest.fixture
def vanilla_memory():
    return VanillaMemory()


@pytest.fixture
def fresh_session(tmp_sessions):
    return Session()
