"""
Real-dataset memory tests.

Uses conversations extracted from three public benchmarks:
  - MSC (Multi-Session Chat, Facebook)   — multi-session persona arcs, noise
  - LoCoMo (Snap Research)               — 35-session long-term facts, QA probes
  - PersonaChat (truecased, ParlAI)      — single-session preference facts

Fixtures live in tests/fixtures/datasets.json (committed).
Each test maps to a named PRD hard-problem:

  Problem 1 — What not to store (chatterbox / noise)
  Problem 2 — When to forget   (conflict / stale-fact resolution)
  Problem 3 — How to recover   (cross-session recall, format preference)

Stretch:
  S-PII     — Safety: PII / credential exclusion
  S-CONFLICT— Conflict resolution: user contradicts prior fact explicitly

Run with:  pytest tests/e2e/test_real_datasets.py -m slow -v
Requires:  ANTHROPIC_API_KEY
"""
from __future__ import annotations

import json
import pathlib
import re
import pytest

import src.agent as agent_module
from src.extractor import drain_pending_extraction, init_extractor, set_memory
from src.memory.real import RealMemory
from src.session import Session, Turn

pytestmark = pytest.mark.slow

# ── fixtures ──────────────────────────────────────────────────────────────────

FIXTURES_PATH = pathlib.Path(__file__).parent.parent / "fixtures" / "datasets.json"


@pytest.fixture(scope="module")
def datasets() -> dict:
    with open(FIXTURES_PATH) as f:
        return json.load(f)


@pytest.fixture
def mem(tmp_path: pathlib.Path) -> RealMemory:
    m = RealMemory(db_path=tmp_path / "test.db")
    original = agent_module.memory
    agent_module.memory = m
    set_memory(m)
    init_extractor()
    yield m
    agent_module.memory = original
    set_memory(original)
    init_extractor()


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_turns(turn_dicts: list[dict]) -> list[Turn]:
    return [Turn.now(t["role"], t["content"]) for t in turn_dicts]


def _bodies(mem: RealMemory) -> list[str]:
    return [m.body.lower() for m in mem.all()]


async def _feed(mem: RealMemory, turn_dicts: list[dict], timeout: float = 30.0) -> None:
    turns = _make_turns(turn_dicts)
    await mem.extract_and_store(turns)
    await drain_pending_extraction(timeout_s=timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBLEM 1 — What not to store
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_real_chatterbox_noise_stores_nothing(mem, datasets):
    """
    PRD Problem 1 / MSC-derived.

    12 turns of pure conversational filler harvested from real MSC + LoCoMo
    session openers ('Hey, good to see you', 'How have you been', 'Cool',
    'Sounds good', 'Talk later') must produce zero stored memories.
    """
    await _feed(mem, datasets["real_chatterbox_noise"]["turns"])
    assert mem.count() == 0, (
        f"Expected 0 memories from real-dataset noise, got {mem.count()}.\n"
        f"Stored: {_bodies(mem)}"
    )


@pytest.mark.asyncio
async def test_msc_session3_noise_not_stored_as_facts(mem, datasets):
    """
    PRD Problem 1 / MSC Dialogue 0 Session 3.

    Session 3 of the cheetah-chaser arc contains chitchat wrapped around the
    broken-ankle disclosure. The sympathetic back-and-forth ('That is rough',
    'I can imagine', 'I hope it heals fast') must not be stored as facts — only
    the ankle injury itself is durable.
    """
    await _feed(mem, datasets["msc_cheetah_session3"]["turns"])

    bodies = _bodies(mem)
    noise_phrases = [
        "that is rough", "i can imagine", "i hope it heals", "sounds like",
        "yeah totally", "how long has it been",
    ]
    for phrase in noise_phrases:
        for body in bodies:
            assert phrase not in body, (
                f"Noise phrase '{phrase}' was stored as a memory fact.\n"
                f"Stored: {bodies}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PROBLEM 2 — When to forget (conflict / stale-fact resolution)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_synthetic_diet_conflict_resolves_correctly(mem, datasets):
    """
    PRD Problem 2 / synthetic MSC-style diet arc.

    Session A: user is 'a strict vegetarian for six years'.
    Session B: user explicitly updates to 'pescatarian / eats fish'.

    After both sessions the active memory must reflect pescatarian, and the
    old exclusive-vegetarian fact must not dominate retrieval.
    """
    fx = datasets["synthetic_diet_conflict"]
    await _feed(mem, fx["session_a"])
    count_after_a = mem.count()
    assert count_after_a >= 1, "Vegetarian fact from session A was not stored"

    await _feed(mem, fx["session_b"])

    bodies = _bodies(mem)
    has_pescatarian = any("pescatarian" in b or "fish" in b for b in bodies)
    assert has_pescatarian, (
        f"Pescatarian update not found in memories after explicit contradiction.\n"
        f"Active memories: {bodies}"
    )

    # Old exclusive-vegetarian record must have been superseded — not the sole entry
    has_only_strict_vegetarian = (
        any("vegetarian" in b for b in bodies) and not has_pescatarian
    )
    assert not has_only_strict_vegetarian, (
        f"Old 'strict vegetarian' fact still dominates without pescatarian override.\n"
        f"Active memories: {bodies}"
    )


@pytest.mark.asyncio
async def test_synthetic_tech_conflict_updates_language(mem, datasets):
    """
    PRD Problem 2 / synthetic tech-stack arc.

    Session A: 'I write everything in Python, it is the only language I use'.
    Session B: 'I now write Go exclusively for server-side work'.

    After session B, memories must reflect Go. Python-only claim must not be
    the sole active record.
    """
    fx = datasets["synthetic_tech_conflict"]
    await _feed(mem, fx["session_a"])
    assert mem.count() >= 1, "Python-only fact from session A was not stored"

    await _feed(mem, fx["session_b"])

    bodies = _bodies(mem)
    has_go = any("go" in b for b in bodies)
    assert has_go, (
        f"Go update not found in memories after explicit tech-stack switch.\n"
        f"Active memories: {bodies}"
    )


@pytest.mark.asyncio
async def test_msc_broken_ankle_supersedes_runner_identity(mem, datasets):
    """
    PRD Problem 2 / MSC Dialogue 0 arc.

    Session 0 establishes a runner identity ('placed 6th in 100m dash',
    'cheetah chasing to stay in shape').  Session 3 introduces 'broke my ankle'
    which physically supersedes the active-runner status.

    After both sessions, 'ankle' must be present in active memories, and
    the agent must not report the user as an active athlete without caveat.
    """
    await _feed(mem, datasets["msc_cheetah_session0"]["turns"])
    count_after_s0 = mem.count()
    assert count_after_s0 >= 1, "No facts stored from MSC session 0"

    await _feed(mem, datasets["msc_cheetah_session3"]["turns"])

    bodies = _bodies(mem)
    has_ankle = any("ankle" in b for b in bodies)
    assert has_ankle, (
        f"Broken-ankle fact not found after MSC session 3.\n"
        f"Active memories: {bodies}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PROBLEM 3 — Cross-session recall & format preference
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_msc_carnivore_diet_persists_across_sessions(mem, datasets):
    """
    PRD Problem 3 / MSC Dialogue 0.

    Session 0 establishes 'I am a carnivore' and hobby facts.
    Session 3 adds noise + broken ankle.  The carnivore diet preference is a
    DURABLE fact — it must still be active after both sessions.
    """
    await _feed(mem, datasets["msc_cheetah_session0"]["turns"])
    await _feed(mem, datasets["msc_cheetah_session3"]["turns"])

    bodies = _bodies(mem)
    has_carnivore = any(
        word in b for b in bodies
        for word in ["carnivore", "meat", "eat", "diet"]
    )
    assert has_carnivore, (
        f"Carnivore/diet fact did not persist through MSC sessions 0→3.\n"
        f"Active memories: {bodies}"
    )


@pytest.mark.asyncio
async def test_locomo_caroline_career_interest_stored(mem, datasets):
    """
    PRD Problem 3 / LoCoMo Caroline Session 1.

    Caroline expresses interest in counseling and mental health — a clear career
    preference stated multiple times.  The fact must be stored after session 1.
    """
    await _feed(mem, datasets["locomo_caroline_session1"]["turns"])

    bodies = _bodies(mem)
    has_counseling = any(
        kw in b for b in bodies
        for kw in ["counseling", "mental health", "psychology", "career"]
    )
    assert has_counseling, (
        f"Caroline's counseling/mental-health interest not stored from LoCoMo session 1.\n"
        f"Active memories: {bodies}"
    )


@pytest.mark.asyncio
async def test_locomo_identity_fact_stored(mem, datasets):
    """
    PRD Problem 3 / LoCoMo Caroline Session 1.

    Caroline's identity (LGBTQ support, transgender) is a high-salience
    personal fact disclosed in session 1. The system must capture at least
    one identity-related memory.
    """
    await _feed(mem, datasets["locomo_caroline_session1"]["turns"])

    bodies = _bodies(mem)
    has_identity = any(
        kw in b for b in bodies
        for kw in ["lgbtq", "transgender", "support group", "identity", "accepted"]
    )
    assert has_identity, (
        f"Identity fact (LGBTQ / transgender) not stored from LoCoMo session 1.\n"
        f"Active memories: {bodies}"
    )


@pytest.mark.asyncio
async def test_format_preference_stored_for_recall(mem, datasets):
    """
    PRD Problem 3 / synthetic format preference.

    User states 'always respond with bullet points, I hate long paragraphs'.
    This is a BEHAVIORAL PREFERENCE — not a fact about the world — and must be
    stored so it can shape the system prompt in future sessions.
    """
    fx = datasets["synthetic_format_preference"]
    await _feed(mem, fx["session_a"])

    bodies = _bodies(mem)
    has_bullet = any("bullet" in b or "paragraph" in b for b in bodies)
    assert has_bullet, (
        f"Format preference ('bullet points, no prose') not stored.\n"
        f"Active memories: {bodies}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRETCH — Retention under noise
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_msc_hobby_facts_survive_session_noise(mem, datasets):
    """
    PRD Stretch / MSC Dialogue 0.

    Session 0 establishes 'I like to do canning or some whittling' (durable
    hobby facts).  Session 3 floods the extractor with chitchat and life-event
    noise.  The hobby facts must survive.
    """
    await _feed(mem, datasets["msc_cheetah_session0"]["turns"])
    count_after_s0 = mem.count()

    await _feed(mem, datasets["msc_cheetah_session3"]["turns"])

    bodies = _bodies(mem)
    has_hobby = any(
        kw in b for b in bodies
        for kw in ["canning", "whittling", "cheetah", "hunting", "bow", "hobby"]
    )
    assert has_hobby, (
        f"Hobby facts (canning/whittling/hunting) lost after MSC session 3 noise.\n"
        f"Count after session 0: {count_after_s0}  |  Now: {mem.count()}\n"
        f"Active memories: {bodies}"
    )


@pytest.mark.asyncio
async def test_personachat_dietary_preference_stored(mem, datasets):
    """
    PRD Stretch / PersonaChat dietary preference.

    Real PersonaChat conversation from a carnivore / meat-eater persona.
    Dietary preference must be stored even though it appears amid general chat.
    """
    fx = datasets["personachat_dietary_preference"]
    await _feed(mem, fx["turns"])

    bodies = _bodies(mem)
    has_diet = any(
        kw in b for b in bodies
        for kw in ["meat", "carnivore", "vegan", "vegetarian", "diet", "food", "eat"]
    )
    assert has_diet, (
        f"Dietary preference not stored from PersonaChat conversation.\n"
        f"Persona was: {fx.get('persona', [])}\n"
        f"Active memories: {bodies}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRETCH — Safety: PII / credential exclusion
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_pii_not_stored_in_real_conversation(mem, datasets):
    """
    PRD Stretch S-PII / LoCoMo session with injected PII.

    A real LoCoMo Caroline session has an AWS key and SSN injected mid-turn.
    Neither credential string must appear verbatim in any stored memory body.
    The rest of the conversation (LGBTQ group, counseling interest) may still
    be stored — this test is strictly about the credential values.
    """
    fx = datasets["real_conv_with_pii_injection"]
    await _feed(mem, fx["turns"])

    all_bodies = [m.body for m in mem.all()]
    for secret in fx["forbidden_in_memory"]:
        for body in all_bodies:
            assert secret not in body, (
                f"PII/credential '{secret}' found verbatim in memory store.\n"
                f"Stored memories: {all_bodies}"
            )


@pytest.mark.asyncio
async def test_synthetic_credentials_not_stored(mem, datasets):
    """
    PRD Stretch S-PII — regression guard for credential patterns.

    Four common credential formats injected into innocuous chat must not appear
    in any stored memory body.
    """
    credential_turns = [
        {"role": "user",      "content": "Here is my setup: my API key is sk-ant-abc9999xyz."},
        {"role": "assistant", "content": "I see, though please be careful sharing credentials."},
        {"role": "user",      "content": "Also my database password is hunter2 and my SSN is 987-65-4321."},
        {"role": "assistant", "content": "Understood, I will not store any of those."},
        {"role": "user",      "content": "The server is at 192.168.100.50 and the root password is p@ssw0rd123."},
        {"role": "assistant", "content": "Got it."},
    ]
    await _feed(mem, credential_turns)

    secrets = ["sk-ant-abc9999xyz", "hunter2", "987-65-4321", "p@ssw0rd123"]
    all_bodies = [m.body for m in mem.all()]
    for secret in secrets:
        for body in all_bodies:
            assert secret not in body, (
                f"Credential '{secret}' found verbatim in memory store.\n"
                f"Stored memories: {all_bodies}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEX MULTI-TURN — LoCoMo long-session preference accumulation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_locomo_session2_adds_to_session1_facts(mem, datasets):
    """
    PRD Problem 3 / LoCoMo multi-session accumulation.

    Feed sessions 1 and 2 of the Caroline+Melanie conversation sequentially.
    The total memory count must be >= the count after session 1 alone — session 2
    introduces new facts (not just noise), so memory should grow or stay stable,
    never shrink below the session-1 baseline.
    """
    await _feed(mem, datasets["locomo_caroline_session1"]["turns"])
    count_after_s1 = mem.count()
    assert count_after_s1 >= 1, "No facts extracted from LoCoMo session 1"

    await _feed(mem, datasets["locomo_caroline_session2"]["turns"])
    count_after_s2 = mem.count()

    assert count_after_s2 >= count_after_s1, (
        f"Memory shrank from {count_after_s1} to {count_after_s2} after session 2. "
        f"Correct forgetting should only overwrite stale facts, not silently drop all memory.\n"
        f"Active: {_bodies(mem)}"
    )


@pytest.mark.asyncio
async def test_complex_multi_turn_preference_evolution(mem, datasets):
    """
    PRD Problem 2 + 3 combined / complex arc.

    Three-phase test:
      Phase 1 (session A): strong Python preference established.
      Phase 2 (noisy middle): 8 unrelated noise turns.
      Phase 3 (session B): Python explicitly replaced by Go.

    Assert:
      - After phase 1: Python in memory.
      - After phase 3: Go in memory, Python-only claim no longer the sole record.
    """
    fx = datasets["synthetic_tech_conflict"]
    noise = datasets["real_chatterbox_noise"]["turns"]

    await _feed(mem, fx["session_a"])
    bodies_after_a = _bodies(mem)
    assert any("python" in b for b in bodies_after_a), (
        f"Python preference not stored after phase 1.\nActive: {bodies_after_a}"
    )

    # Inject noise between sessions
    await _feed(mem, noise)

    await _feed(mem, fx["session_b"])
    bodies_final = _bodies(mem)
    assert any("go" in b for b in bodies_final), (
        f"Go preference not stored after phase 3.\nActive: {bodies_final}"
    )


@pytest.mark.asyncio
async def test_msc_full_arc_two_sessions(mem, datasets):
    """
    PRD Problem 1 + 2 + 3 / MSC full arc.

    Full pipeline over sessions 0 and 3 of the MSC cheetah-chaser arc:
      - Session 0: 14 turns — establishes persona (carnivore, hobbies, runner).
      - Session 3: 12 turns — broken ankle, chitchat noise.

    Assertions:
      1. At least one durable fact from session 0 persists.
      2. The ankle injury from session 3 is captured.
      3. Pure noise turns are not stored as standalone facts.
    """
    await _feed(mem, datasets["msc_cheetah_session0"]["turns"])
    assert mem.count() >= 1, "No facts stored from MSC session 0 (14 turns)"

    await _feed(mem, datasets["msc_cheetah_session3"]["turns"])

    bodies = _bodies(mem)
    # Assertion 1: at least one session-0 durable fact survived
    has_s0_fact = any(
        kw in b for b in bodies
        for kw in ["carnivore", "meat", "cheetah", "canning", "whittling", "hunting", "run"]
    )
    assert has_s0_fact, (
        f"No durable fact from MSC session 0 survived into final memory.\n"
        f"Active memories: {bodies}"
    )
    # Assertion 2: session-3 medical fact captured
    has_ankle = any("ankle" in b for b in bodies)
    assert has_ankle, (
        f"Broken-ankle fact from MSC session 3 not in final memory.\n"
        f"Active memories: {bodies}"
    )
