import pathlib

from .adjudicator import adjudicate
from .models import CandidateFact
from .observer import observe
from .retriever import retrieve as _retrieve
from .store import MemoryStore
from .stub import Memory

_DEFAULT_DB = pathlib.Path("~/.agent/memories/memories.db").expanduser()


class RealMemory:
    """
    Persistent memory store. Satisfies MemoryInterface.

    Read path  (sync, ~25ms): embed query → cosine top-K → list[Memory]
    Write path (async, background):
        turns → observer (extract durable facts)
              → adjudicator (ADD / UPDATE / DELETE / NONE)
              → SQLite store
    """

    def __init__(self, db_path: str | pathlib.Path = _DEFAULT_DB) -> None:
        self._store = MemoryStore(db_path)

    # ── MemoryInterface ──────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[Memory]:
        return _retrieve(query, self._store, k=8)

    async def extract_and_store(self, turns: list) -> None:
        pairs = _pair_turns(turns)
        for i, (user_text, assistant_text) in enumerate(pairs):
            prior = _build_prior_context(pairs, i)
            candidates: list[CandidateFact] = await observe(user_text, assistant_text, prior_context=prior)
            for candidate in candidates:
                await adjudicate(candidate, self._store)

    # ── optional introspection ───────────────────────────────────────────────

    def count(self) -> int:
        return self._store.count_active()

    def all(self) -> list[Memory]:
        """Return all active memories — useful for /memories inspect command."""
        from .retriever import _to_memory
        return [_to_memory(r) for r in self._store.all_active()]


def _build_prior_context(pairs: list[tuple[str, str]], current_idx: int, window: int = 2) -> str:
    """Return up to `window` prior turn pairs as a mini-transcript for the observer."""
    prior = pairs[max(0, current_idx - window):current_idx]
    if not prior:
        return ""
    lines = []
    for u, a in prior:
        lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a[:120]}{'...' if len(a) > 120 else ''}")
    return "\n".join(lines)


def _pair_turns(turns: list) -> list[tuple[str, str]]:
    """
    Convert a flat list of Turn objects into (user_text, assistant_text) pairs.
    Handles odd-length sequences: unpaired user turn gets empty assistant string.
    """
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(turns):
        if turns[i].role == "user":
            user_text = turns[i].content
            assistant_text = ""
            if i + 1 < len(turns) and turns[i + 1].role == "assistant":
                assistant_text = turns[i + 1].content
                i += 2
            else:
                i += 1
            pairs.append((user_text, assistant_text))
        else:
            i += 1
    return pairs
