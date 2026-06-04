"""
PRD Stress: Chatterbox & Hypothetical Filtering
Target: PRD "What Not to Store" — observer must silence noise, keep signal.

Harder than v1: includes roleplay, third-party attribution, book quotes,
conditional hypotheticals, vague intent, and negations — all interspersed
with real durable facts the observer MUST capture.

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


async def main() -> int:
    db_path = pathlib.Path("~/.agent/memories/prd_chatterbox_v2.db").expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        os.remove(db_path)

    mem = RealMemory(db_path=db_path)
    agent_module.memory = mem
    set_memory(mem)
    init_extractor()
    s = Session()

    turns = [
        # ── Pure hypotheticals — must NOT be stored ──────────────────────
        "If I were to move to Mars, I think I'd really miss the Pacific Ocean.",
        "Suppose I had become a marine biologist instead, I'd work with dolphins all day.",
        "If I lost my job tomorrow, I'd probably move back to my hometown.",

        # ── Roleplay framing — must NOT store roleplay persona ────────────
        "Let's do a roleplay. For the next few messages, I'm an airline pilot named Captain Torres.",
        "As Captain Torres, my home base is Dallas/Fort Worth International Airport.",
        # Snap back to real self
        "Okay, roleplay over. In real life I'm a backend engineer at a fintech startup.",

        # ── Third-party attribution — must NOT store as user's own facts ──
        "My colleague Priya uses Vim and swears by it, but personally I find it too steep.",
        "My mum is a retired teacher who loves gardening. I myself don't garden at all.",
        "I was reading a sci-fi novel where the protagonist's favourite OS was Haiku OS.",

        # ── Conditional / vague / uncertain — must NOT be stored ─────────
        "I might start learning Rust someday, though I haven't committed to it yet.",
        "I'm thinking about going vegan, not sure. Still eating meat for now.",

        # ── Book and movie quotes — must NOT be stored as user facts ──────
        'The character in the book I finished said: "My life\'s purpose is to travel every continent."',
        "In that Netflix show the protagonist hates coffee — the opposite of me actually.",

        # ── Negations — negative facts are borderline; must not store as positives ──
        "I don't drink alcohol at all.",
        "I've never owned a pet and never plan to.",

        # ── REAL DURABLE FACTS — must be stored ──────────────────────────
        "Anyway, my actual name is Rajan Mehta.",
        "I'm a backend engineer with 6 years of experience, currently at FinEdge Ltd.",
        "I absolutely love sushi — it's my go-to meal for any celebration.",
        "My primary language at work is Go, but I also write a lot of Python for tooling.",
        "My long-term goal is to become a staff engineer within the next two years.",
    ]

    print("=" * 70)
    print("PRD STRESS: CHATTERBOX & HYPOTHETICAL FILTERING")
    print("=" * 70)
    for msg in turns:
        print(f"\n\033[1mUser:\033[0m {msg}")
        response, _ = await chat(msg, s)
        print(f"\033[90mAgent:\033[0m {response[:120]}{'...' if len(response) > 120 else ''}")

    print("\n[Draining extraction...]")
    await drain_pending_extraction(timeout_s=30.0)

    memories = mem.all()
    bodies = [m.body.lower() for m in memories]
    all_bodies = " ".join(bodies)

    print(f"\n{'=' * 70}")
    print(f"STORED MEMORIES ({len(memories)} records):")
    for m in memories:
        print(f"  [{m.type}] {m.body}")

    # ── Invariant checks ─────────────────────────────────────────────────
    checks: list[tuple[str, bool, str]] = []

    def chk(label: str, condition: bool, detail: str = "") -> None:
        checks.append((label, condition, detail))

    # Must-NOT-store (noise/hypotheticals)
    chk("No Mars hypothetical stored",    "mars" not in all_bodies)
    chk("No marine biologist stored",     "marine biologist" not in all_bodies)
    chk("No roleplay persona stored",     "captain torres" not in all_bodies)
    chk("No DFW airport stored",          "dallas" not in all_bodies and "fort worth" not in all_bodies)
    chk("No Priya's Vim preference",      "priya" not in all_bodies)
    chk("No mum's gardening stored",      "gardening" not in all_bodies and "mum" not in all_bodies)
    chk("No sci-fi novel OS stored",      "haiku os" not in all_bodies)
    chk("No vague Rust intent stored",    "rust" not in all_bodies or "might" not in all_bodies.split("rust")[0][-30:])
    chk("No book quote travel goal",      "travel every continent" not in all_bodies)

    # Must-STORE (durable real facts)
    chk("Name Rajan Mehta stored",        any("rajan" in b or "mehta" in b for b in bodies))
    chk("Role backend engineer stored",   any("backend" in b for b in bodies))
    chk("Employer FinEdge stored",        any("finedge" in b for b in bodies))
    chk("Sushi preference stored",        any("sushi" in b for b in bodies))
    chk("Go language stored",             any(" go" in b or "golang" in b for b in bodies))
    chk("Staff engineer goal stored",     any("staff engineer" in b for b in bodies))

    print(f"\n{'=' * 70}")
    print("INVARIANT CHECKS:")
    failures = 0
    for label, passed, detail in checks:
        status = PASS if passed else FAIL
        suffix = f" — {detail}" if detail else ""
        print(f"  {status}  {label}{suffix}")
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
