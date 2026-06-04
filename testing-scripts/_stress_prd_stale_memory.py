"""
PRD Stress: Evolving & Stale Memory — Chain Updates Across Sessions
Target: PRD "Conflict resolution" + "how to recover when a memory is wrong or stale"

Harder than v1: 6 sessions with cascading contradictions.
Each session prints a per-session state snapshot, and the final section
runs automated PASS/FAIL invariant checks against the expected end-state.

Session chain:
  1  Baseline:   junior FE dev, Windows, Python 3.9, Chicago, single
  2  Promotion:  mid-level FE dev, dual-boot Windows/Linux, Python 3.11
  3  Job change: moved to a different startup, same FE role
  4  Role pivot: dropped FE entirely → DB admin (hard contradiction)
  5  Relocation: moved from Chicago to Berlin, switched to Macbook exclusively
  6  New aspiration: enrolled in part-time PhD, in a relationship

Exit code 0 = PASS  /  Exit code 1 = FAIL
"""

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

import src.agent as agent_module
from src.agent import chat
from src.extractor import drain_pending_extraction, init_extractor, set_memory
from src.memory.real import RealMemory
from src.session import Session

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

SESSIONS = [
    # (session_num, description, turns)
    (1, "Baseline: junior FE dev, Windows, Python 3.9, Chicago, single", [
        "I'm a junior frontend developer at a small agency called PixelCraft.",
        "I exclusively use Windows 11 as my OS.",
        "My go-to language for scripting is Python 3.9.",
        "I live in Chicago, Illinois.",
        "I'm currently single.",
    ]),
    (2, "Promotion & stack upgrade (UPDATE expected, not duplicate)", [
        "Great news — I got promoted to mid-level frontend dev last week.",
        "I set up a dual-boot: Windows 11 and Ubuntu 22.04 now.",
        "I upgraded to Python 3.11 for all my projects.",
    ]),
    (3, "Company change — same role, new employer (UPDATE expected)", [
        "I left PixelCraft and joined a bigger startup called Nexora.",
        "Still doing frontend, but the codebase is React with TypeScript.",
    ]),
    (4, "Hard role pivot — FE career dropped, full DBA (DELETE + ADD expected)", [
        "Big career change: I've completely dropped frontend development.",
        "I transitioned to a database administrator role at Nexora.",
        "My daily tools are now PostgreSQL and Redis, no more JavaScript.",
    ]),
    (5, "Relocation + hardware switch (DELETE + ADD expected)", [
        "I moved from Chicago to Berlin last month for work.",
        "I sold my Windows machine. I now use a MacBook Pro M3 exclusively.",
        "Linux dual-boot is gone — just macOS now.",
    ]),
    (6, "New life goals + relationship (ADD expected, no conflicts)", [
        "I enrolled in a part-time PhD programme in distributed systems.",
        "I've been in a relationship for three months now.",
        "My goal is to finish the PhD within five years while staying employed.",
    ]),
]


async def run_session(num: int, desc: str, turns: list[str], mem: RealMemory) -> None:
    print(f"\n{'─' * 70}")
    print(f"SESSION {num}: {desc}")
    print("─" * 70)
    init_extractor()
    s = Session()
    for msg in turns:
        print(f"  User: {msg}")
        await chat(msg, s)
    await drain_pending_extraction(timeout_s=25.0)
    print(f"\n  → State after session {num} ({mem.count()} active records):")
    for m in mem.all():
        print(f"      [{m.type}] {m.body}")


async def main() -> int:
    db_path = pathlib.Path("~/.agent/memories/prd_stale_v2.db").expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        os.remove(db_path)

    mem = RealMemory(db_path=db_path)
    agent_module.memory = mem
    set_memory(mem)

    print("=" * 70)
    print("PRD STRESS: EVOLVING & STALE MEMORY — 6-SESSION CHAIN")
    print("=" * 70)

    for num, desc, turns in SESSIONS:
        await run_session(num, desc, turns, mem)

    final = mem.all()
    bodies = [m.body.lower() for m in final]
    all_bodies = " ".join(bodies)

    print(f"\n{'=' * 70}")
    print(f"FINAL STATE ({len(final)} active records):")
    for m in final:
        print(f"  [{m.type}] {m.body}")

    # ── Invariant checks ─────────────────────────────────────────────────
    checks: list[tuple[str, bool]] = []

    def chk(label: str, condition: bool) -> None:
        checks.append((label, condition))

    # Role evolution — final role must be DBA, not frontend
    chk("Final role is DBA",           any("database administrator" in b or "dba" in b for b in bodies))
    chk("Frontend role removed",       not any("frontend developer" in b and "junior" in b for b in bodies))
    chk("Junior title removed",        not any("junior" in b for b in bodies))

    _NEGATIONS = ("no longer", "not ", "sold", "exclusively mac", "dropped", "no more", "removed")

    def _positive_claim(body: str, keyword: str) -> bool:
        """True only if keyword appears WITHOUT a negation modifier."""
        return keyword in body and not any(neg in body for neg in _NEGATIONS)

    # OS chain — final OS must be macOS only
    chk("MacBook / macOS present",     any("macbook" in b or "macos" in b or "mac" in b for b in bodies))
    chk("Windows exclusive removed",   not any(_positive_claim(b, "windows") for b in bodies))
    chk("Dual-boot removed",           not any(_positive_claim(b, "dual-boot") or _positive_claim(b, "ubuntu") for b in bodies))

    # Location — final location must be Berlin, not Chicago
    chk("Berlin present",              any("berlin" in b for b in bodies))
    chk("Chicago as current removed",  not any(b.strip().endswith("chicago, illinois.") or b.strip().endswith("chicago.") for b in bodies))

    # Employer chain — should reflect Nexora, not PixelCraft as current
    chk("Nexora present",              any("nexora" in b for b in bodies))
    chk("PixelCraft as active removed", not any("pixelcraft" in b and "left" not in b for b in bodies))

    # Skills — no more frontend/JS tools
    chk("PostgreSQL/Redis present",    any("postgresql" in b or "redis" in b for b in bodies))
    chk("JavaScript as primary removed", not any(_positive_claim(b, "javascript") for b in bodies))

    # New additions from sessions 5 & 6
    chk("PhD goal stored",             any("phd" in b or "doctoral" in b for b in bodies))
    chk("Relationship status stored",  any("relationship" in b for b in bodies))

    # No unbounded duplication — record count should be reasonable
    chk("Record count reasonable (≤20)", len(final) <= 20)

    print(f"\n{'=' * 70}")
    print("INVARIANT CHECKS:")
    failures = 0
    for label, passed in checks:
        status = PASS if passed else FAIL
        print(f"  {status}  {label}")
        if not passed:
            failures += 1

    print(f"\n{'=' * 70}")
    if failures == 0:
        print(f"VERDICT: {PASS} — all {len(checks)} invariants satisfied")
    else:
        print(f"VERDICT: {FAIL} — {failures}/{len(checks)} invariants violated")
    print("=" * 70)
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
