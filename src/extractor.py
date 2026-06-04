"""
Coalesced, cursor-based background memory extraction.

Key properties:
  - Only one extraction runs at a time (no concurrent pile-up).
  - If a new turn arrives while extraction is running, the session is stashed
    and one trailing run processes it after the current run finishes.
  - Cursor (_last_processed_uuid) ensures each run only processes new turns.
  - drain_pending_extraction() lets the CLI wait for the last run before exit.
"""

import asyncio
from .session import Session

# All mutable state is module-level so init_extractor() can fully reset it.
# Call init_extractor() at session start and in test setUp/teardown.
_in_progress: bool = False
_pending_session: Session | None = None
_in_flight: set[asyncio.Task] = set()
_last_processed_uuid: str | None = None

# Imported lazily to avoid circular imports; agent.py sets this at startup.
_memory = None


def set_memory(memory) -> None:
    """Called once by agent.py to inject the memory instance."""
    global _memory
    _memory = memory


def init_extractor() -> None:
    """Reset all state. Call at session start and in test fixtures."""
    global _in_progress, _pending_session, _in_flight, _last_processed_uuid
    _in_progress = False
    _pending_session = None
    _in_flight = set()
    _last_processed_uuid = None


def schedule_extraction(session: Session) -> None:
    """
    Fire-and-forget: called after each turn, never blocks.
    Coalesces concurrent calls — only one extraction runs at a time.
    """
    global _pending_session, _in_progress
    if _in_progress:
        _pending_session = session  # overwrite: only latest context matters
        return
    # Claim the lock BEFORE creating the task so subsequent synchronous calls
    # to schedule_extraction (before the event loop yields) see _in_progress=True
    # and stash rather than spawning additional tasks.
    _in_progress = True
    task = asyncio.create_task(_run_extraction(session))
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)


async def _run_extraction(session: Session) -> None:
    global _pending_session, _last_processed_uuid, _in_progress
    # _in_progress is already True when we arrive here (set by schedule_extraction).
    try:
        new_turns = _turns_since(session, _last_processed_uuid)
        if new_turns and _memory is not None:
            await _memory.extract_and_store(new_turns)
            # Use new_turns[-1].id, NOT session.history[-1].id.
            # session.history may grow while extract_and_store awaits the LLM,
            # causing the cursor to jump past turns that were never processed.
            _last_processed_uuid = new_turns[-1].id
    except Exception as e:
        print(f"[extraction error] {e}", flush=True)  # never propagates
    finally:
        pending = _pending_session
        _pending_session = None
        if pending:
            # Stay "in progress" — process trailing session immediately.
            await _run_extraction(pending)
        else:
            _in_progress = False  # release only when there's nothing left


async def drain_pending_extraction(timeout_s: float = 30.0) -> None:
    """
    Wait for all in-flight extractions before process exit.
    Gives the background extractor up to timeout_s to finish its run so the
    last session's memories are not silently dropped on /quit or ctrl-D.
    """
    if not _in_flight:
        return
    await asyncio.wait_for(
        asyncio.gather(*_in_flight, return_exceptions=True),
        timeout=timeout_s,
    )


def _turns_since(session: Session, since_id: str | None) -> list:
    """Return only turns added after since_id (cursor-based)."""
    if since_id is None:
        return list(session.history)
    ids = [t.id for t in session.history]
    if since_id not in ids:
        return list(session.history)  # cursor lost — re-process all
    idx = ids.index(since_id)
    return session.history[idx + 1 :]
