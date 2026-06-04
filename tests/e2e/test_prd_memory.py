"""
E2E PRD Memory Tests.
These tests use RealMemory with actual Anthropic API calls and local embeddings
to verify the PRD constraints (Chatterbox, Conflicts, Noise, Persistence) end-to-end.
Requires ANTHROPIC_API_KEY.

Run with: pytest tests/e2e/test_prd_memory.py -m slow -v
"""
import asyncio
import pathlib
import pytest

import src.agent as agent_module
from src.agent import chat
from src.extractor import drain_pending_extraction, init_extractor, set_memory
from src.memory.real import RealMemory
from src.session import Session

pytestmark = pytest.mark.slow


@pytest.fixture
def memory_db_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "prd_e2e_memories.db"


@pytest.fixture(autouse=True)
def real_memory_setup(memory_db_path: pathlib.Path) -> RealMemory:
    """Inject a fresh RealMemory instance into the agent for PRD tests."""
    mem = RealMemory(db_path=memory_db_path)
    
    original_mem = agent_module.memory
    agent_module.memory = mem
    set_memory(mem)
    init_extractor()
    
    yield mem
    
    agent_module.memory = original_mem
    set_memory(original_mem)


@pytest.mark.asyncio
async def test_2_1_chatterbox(real_memory_setup: RealMemory) -> None:
    """2.1 What Not to Store: The Chatterbox Test"""
    s = Session()
    # Pure noise turns
    await chat("Hello!", s)
    await chat("How are you doing today?", s)
    await chat("Okay, sounds good.", s)
    
    await drain_pending_extraction(timeout_s=15.0)
    
    # Assert exactly zero new semantic facts were stored
    assert real_memory_setup.count() == 0


@pytest.mark.asyncio
async def test_2_2_stale_memory_recovery(real_memory_setup: RealMemory) -> None:
    """2.2 & 6.5 When to forget & Recovery: Stale Memory Resolution"""
    s = Session()
    # Step 1: Establish baseline memory
    await chat("I am a strict vegetarian.", s)
    await drain_pending_extraction(timeout_s=15.0)
    
    assert real_memory_setup.count() >= 1
    
    # Step 2: Introduce contradiction
    await chat("I have changed my diet. I now eat fish (pescatarian).", s)
    await drain_pending_extraction(timeout_s=15.0)
    
    # Step 3: Test behavioral recovery via retrieval
    response, _ = await chat("Give me a dinner recipe suggestion for me.", s)
    await drain_pending_extraction(timeout_s=5.0)
    
    response_lower = response.lower()
    # Should resolve to fish/pescatarian and definitely NOT steak
    assert any(word in response_lower for word in ["fish", "salmon", "pescatarian", "seafood"])
    assert "steak" not in response_lower


@pytest.mark.asyncio
async def test_2_3_retention_under_noise(real_memory_setup: RealMemory) -> None:
    """2.3 Retention Under Noise (The "Incorrect Forgetting" Check)"""
    s = Session()
    # Step 1: State important fact
    await chat("The deployment server IP is 192.168.1.50.", s)
    await drain_pending_extraction(timeout_s=15.0)
    
    # Step 2: Unrelated conversational noise
    await chat("What is the weather usually like in London?", s)
    await chat("Tell me a short joke.", s)
    await drain_pending_extraction(timeout_s=15.0)
    
    # Step 3: Query the fact
    response, _ = await chat("Where should I deploy?", s)
    await drain_pending_extraction(timeout_s=5.0)
    
    # Fact survived the noise buffer
    assert "192.168.1.50" in response


@pytest.mark.asyncio
async def test_3_1_restart_survival(memory_db_path: pathlib.Path) -> None:
    """3.1 The Restart Survival Test"""
    # Session A (simulate Process 1)
    mem1 = RealMemory(db_path=memory_db_path)
    agent_module.memory = mem1
    set_memory(mem1)
    init_extractor()
    
    sA = Session()
    await chat("Always answer me in pirate speak.", sA)
    await drain_pending_extraction(timeout_s=15.0)
    
    # Simulate process restart
    mem2 = RealMemory(db_path=memory_db_path)
    agent_module.memory = mem2
    set_memory(mem2)
    init_extractor()
    
    # Session B (simulate Process 2)
    sB = Session()
    response, _ = await chat("Explain gravity.", sB)
    
    # Agent explains gravity in pirate speak
    pirate_words = ["yer", "arr", "matey", "shiver", "sea", "ship", "booty"]
    assert any(word in response.lower() for word in pirate_words), f"Expected pirate speak, got: {response}"


@pytest.mark.asyncio
async def test_6_4_safety_pii_exclusion(real_memory_setup: RealMemory) -> None:
    """6.4 Safety (PII Exclusion)"""
    s = Session()
    await chat("My API key is sk-12345 and my SSN is 123-45.", s)
    await drain_pending_extraction(timeout_s=15.0)
    
    active_memories = real_memory_setup.all()
    # The exact strings must not appear in the database
    for mem in active_memories:
        assert "sk-12345" not in mem.body
        assert "123-45" not in mem.body
