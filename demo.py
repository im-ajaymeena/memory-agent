"""
Demo: memory persists and is recalled across sessions.

Session 1 — user shares facts about themselves.
            Background extraction writes them to ~/.agent/memories/memories.db.
Session 2 — brand-new process start. Agent retrieves stored memories and
            references them without being told anything in this session.

Usage:
    python demo.py                           # two fresh sessions end-to-end
    python demo.py --skip-session-1          # jump to session 2 only
    python demo.py --session-1-id <uuid>     # replay an existing session 1
"""
import argparse
import asyncio

from dotenv import load_dotenv
load_dotenv()

from src.agent import chat, memory
from src.extractor import drain_pending_extraction, init_extractor
from src.session import Session

DIVIDER = "─" * 60


def _header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


async def session_1(session_id: str | None) -> str:
    """User shares preferences — observer extracts them, adjudicator stores them."""
    _header("SESSION 1  —  establishing facts")
    init_extractor()
    s = Session(session_id).load()
    print(f"Session ID : {s.session_id}")
    print(f"Store size : {memory.count()} memories before this session\n")

    exchanges = [
        "Hi! I'm a backend engineer and I almost always write Python.",
        "I'm currently refactoring our auth service — it's a Django monolith.",
        "My main goal right now is reducing p99 latency on our API.",
    ]

    for msg in exchanges:
        print(f"\n> {msg}\n")
        await chat(msg, s)
        print()

    print("[waiting for background extraction to complete...]")
    await drain_pending_extraction(timeout_s=30.0)
    print(f"Store size : {memory.count()} memories after this session")
    print(f"\n[Session 1 complete — session ID: {s.session_id}]")
    return s.session_id


async def session_2() -> None:
    """Fresh session — agent uses retrieved memories without being told anything."""
    _header("SESSION 2  —  cross-session recall (new process)")
    init_extractor()
    s = Session()  # brand-new session ID
    print(f"Session ID : {s.session_id}  (new — no shared history)")
    print(f"Store size : {memory.count()} memories available\n")

    queries = [
        "What do you know about me?",
        "Any quick wins you'd suggest for my current project?",
    ]

    for msg in queries:
        print(f"\n> {msg}\n")
        await chat(msg, s)
        print()

    await drain_pending_extraction(timeout_s=10.0)
    print(f"\n[Session 2 complete]")
    print(f"To inspect stored memories: python -c \"from src.agent import memory; [print(m) for m in memory.all()]\"")


async def main() -> None:
    p = argparse.ArgumentParser(description="Cross-session memory demo")
    p.add_argument("--skip-session-1", action="store_true")
    p.add_argument("--session-1-id", help="Resume existing session 1 by ID")
    args = p.parse_args()

    print("\nMemory Persistence Demo")
    print("=======================")
    print(f"LLM   : claude-sonnet-4-6  (conversations)")
    print(f"Memory: claude-haiku-4-5   (extraction) + fastembed ONNX (retrieval)")
    print(f"Store : SQLite  ~/.agent/memories/memories.db\n")

    if not args.skip_session_1:
        await session_1(args.session_1_id)

    await session_2()


if __name__ == "__main__":
    asyncio.run(main())
