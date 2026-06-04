import asyncio
import time

import anthropic

from .memory.real import RealMemory
from .memory.stub import Memory
from .session import Session, Turn
from . import extractor as _extractor

client = anthropic.AsyncAnthropic()
memory = RealMemory()

# Wire memory into the extractor once at module load.
_extractor.set_memory(memory)

MAX_TURNS_VERBATIM = 20


async def chat(
    user_input: str,
    session: Session,
    abort: asyncio.Event | None = None,
) -> tuple[str, float]:
    """
    Stream a response and return (full_text, ttft_seconds).

    Memory retrieval runs concurrently with history serialization so its
    latency is hidden behind model prefill rather than added to TTFT.
    """
    # 1. Kick off memory retrieval NOW, before the API call.
    memory_prefetch = asyncio.create_task(
        asyncio.to_thread(memory.retrieve, user_input)
    )

    # 2. Build messages while prefetch runs in the background.
    messages = _build_messages(session, user_input)

    # 3. Await prefetch — by now it's almost certainly done.
    memories = await memory_prefetch
    system = _build_system_prompt(memories)

    # 4. Stream with TTFT measurement.
    full_text: list[str] = []
    ttft: float | None = None
    t_send = time.perf_counter()

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            if abort and abort.is_set():
                break
            if ttft is None:
                ttft = time.perf_counter() - t_send
            full_text.append(text)
            print(text, end="", flush=True)

    response = "".join(full_text)

    # 5. Persist turn (crash-safe, sync).
    session.append(Turn.now("user", user_input))
    session.append(Turn.now("assistant", response))

    # 6. Schedule background extraction — coalesced, never blocks the next turn.
    _extractor.schedule_extraction(session)

    return response, ttft or 0.0


def _build_messages(session: Session, user_input: str) -> list[dict]:
    recent = session.history[-MAX_TURNS_VERBATIM:]
    messages = [{"role": t.role, "content": t.content} for t in recent]
    messages.append({"role": "user", "content": user_input})
    return messages


def _build_system_prompt(memories: list[Memory]) -> str:
    base = (
        "You are a helpful assistant. "
        "You remember facts about this user across sessions."
    )
    if not memories:
        return base
    facts = "\n".join(
        f"- [{m.type} | {m.age_human_readable}] {m.body}" for m in memories
    )
    return f"{base}\n\nKnown facts about this user:\n{facts}"
