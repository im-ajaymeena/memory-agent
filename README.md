# Conversational Agent with Persistent Memory

A conversational agent that remembers facts about the user across sessions — built directly on the Anthropic SDK with no agent frameworks.

---

## Quick Start

**Requires Python 3.12+**

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# edit .env — add your ANTHROPIC_API_KEY

# 4. Run
make run
```

The fastembed embedding model (~67 MB) downloads automatically on first run to `~/.cache/fastembed/`. To pre-warm it before your first conversation:
```bash
python -c "from src.memory.embedder import embed; embed('warmup')"
```

**Resume a prior session:**
```bash
python -m src.cli --session-id <uuid>
# or: make sessions  (lists recent session IDs)
```

---

## How It Works

```
User input
    │
    ├─► memory.retrieve(query)        ← embedding cosine search, ~20ms
    │       runs concurrently with      hidden behind model prefill
    │       history serialization
    │
    ├─► _build_messages(session)      ← rolling 20-turn window
    │
    └─► claude-sonnet-4-6 (stream)
            │
            ├─► first token → TTFT recorded
            └─► full response → session.append() → schedule_extraction()
                                                          │
                                                    background task
                                                    (coalesced, cursor-based)
                                                          │
                                              claude-haiku-4-5 observer
                                              extracts durable facts
                                                          │
                                              claude-haiku-4-5 adjudicator
                                              ADD / UPDATE / DELETE / NONE
                                                          │
                                              SQLite store
```

### Memory Architecture

Two independent stores:

**Session store** — `~/.agent/sessions/<uuid>.jsonl`
Append-only JSONL, flushed on every turn. Crash-safe by design; replay on load reconstructs the full conversation.

**Long-term memory store** — `~/.agent/memories/memories.db`
SQLite with 384-dim embeddings (fastembed `BAAI/bge-small-en-v1.5`, local ONNX). Five typed categories (IMDMR schema): `personal_information`, `professional_details`, `preferences_interests`, `goals_aspirations`, `contextual_information`.

### What Gets Stored — and What Doesn't

The observer (claude-haiku) reads each turn and extracts only facts that help the agent be more useful in a future session.

**Stored:** stated preferences, job/role, active projects, explicit goals, personal context.

**Never stored:** greetings, filler, temporary task state ("fix bug on line 42"), code snippets, credentials/API keys/passwords (explicit exclusion in the observer prompt), facts stated by the assistant (only user-sourced facts are extracted).

The adjudicator (claude-haiku) prevents duplicates by deciding ADD / UPDATE / DELETE / NONE before every write. A fast-path skips the LLM call when a near-identical fact (cosine > 0.92) already exists at equal or higher trust.

### Latency — Why Memory Doesn't Slow Down Responses

**Concurrent prefetch** — `memory.retrieve()` fires as an `asyncio.Task` before the API call, running in a thread so the ~20ms embed + cosine search is hidden behind model prefill.

**Background extraction** — `extract_and_store()` runs after the response is returned. The user sees the reply immediately; extraction runs concurrently.

**Coalesced extraction** — only one extraction runs at a time. Rapid turns stash the latest session; one trailing run catches up, preventing pile-up of concurrent LLM calls. Cursor-based (`_last_processed_uuid`) so each run only sees new turns.

**Drain before exit** — `drain_pending_extraction(timeout_s=30)` is called on `/quit` or ctrl-D so the last session's extractions aren't silently dropped.

---

## Design Tradeoffs

| Decision | What was traded away | Why |
|---|---|---|
| Haiku for extraction + adjudication | Sonnet-quality reasoning | Extraction is a structured JSON task; Haiku is fast and cheap enough for background work |
| Local ONNX embeddings (fastembed) | Cloud embedding API | ~5–15ms with no network round-trip; zero per-call cost; no extra API key |
| SQLite | PostgreSQL / dedicated vector DB | Self-contained, zero ops, sufficient for ≤5K memories |
| Cosine O(N) scan | ANN index (FAISS/hnswlib) | O(N) numpy matmul is fast enough at 1K; ANN needed beyond ~5K |
| Append-only JSONL sessions | SQLite sessions | Sessions are a sequential log; JSONL is crash-safe, human-readable, zero schema |
| Rolling 20-turn history window | Full context | Prevents token-limit hits; summarization is the natural next step |

---

## What I'd Build Next

1. **Conversation compaction** — summarize turns older than the rolling window into a single context entry rather than dropping them.
2. **ANN index** — FAISS or hnswlib for retrieval beyond 5K memories.
3. **Implicit staleness detection** — currently only explicit supersession language ("I switched from X to Y") triggers DELETE. Gradual drift ("I've been writing more Go lately") doesn't.
4. **Memory confidence decay** — reduce trust score of old facts over time; surface low-confidence memories with a caveat.

---

## Project Structure

```
src/
  agent.py            # chat() — streaming, TTFT, memory prefetch
  extractor.py        # coalesced background extraction + drain
  session.py          # Session + Turn — append-only JSONL
  cli.py              # interactive REPL (prompt_toolkit)
  memory/
    real.py           # RealMemory — full implementation
    stub.py           # Memory dataclass + VanillaMemory (for tests)
    observer.py       # claude-haiku: extract durable facts from turns
    adjudicator.py    # claude-haiku: ADD / UPDATE / DELETE / NONE
    embedder.py       # fastembed BAAI/bge-small-en-v1.5 ONNX
    retriever.py      # embed query → cosine top-K → Memory objects
    store.py          # SQLite CRUD + cosine search
    models.py         # MemoryRecord, CandidateFact, Category, Source
    interface.py      # MemoryInterface Protocol

tests/
  unit/               # pure functions, no I/O (instant, no API key needed)
  integration/        # mocked LLM boundary, real session/memory
  e2e/                # real LLM calls — requires ANTHROPIC_API_KEY
  memory/             # memory module unit + integration tests
  benchmark/          # latency SLA assertion
  fixtures/           # shared datasets (MSC, LoCoMo, PersonaChat)

testing-scripts/
  benchmark.py             # p50 TTFT at 0 vs 1K memories — asserts <200ms delta
  populate_store.py        # inject N synthetic memories for benchmarking
  _stress_prd_categories_pii.py   # PII safety + category classification stress
  _stress_prd_chatterbox.py       # noise filtering: roleplay, quotes, hypotheticals
  _stress_prd_stale_memory.py     # 6-session cascading conflict resolution
  _stress_test_cli.py             # rapid-fire contradictions across sessions
  _stress_test_large_noise.py     # signal survival in large noise

demo.py               # two-session cross-memory demonstration
```

---

## Running Tests

```bash
make test            # unit + integration — no API key needed (~2s)
make test-e2e        # real LLM calls — requires ANTHROPIC_API_KEY (~60s)
make test-benchmark  # latency SLA: p50 delta must be <200ms (~5 min)
make benchmark       # standalone latency benchmark with detailed output
```

---

## Demo

```bash
make demo
```

Runs two sessions back-to-back:
1. User tells the agent their role, current project, and goal.
2. A fresh session starts (new session ID, simulating a process restart). The agent recalls the facts from session 1 without being told anything.

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Conversations (claude-sonnet-4-6) + extraction (claude-haiku-4-5) |

---

## Known Limitations

- **Implicit staleness** — gradual fact drift without explicit contradiction isn't detected.
- **O(N) retrieval** — `store.search()` loads all active rows on every call. Fast at ≤5K; needs ANN beyond that.

---

## Time Spent

| Area | Time |
|---|---|
| Architecture research + design | ~1.5h |
| Agent core + session + extractor + CLI | ~1.5h |
| Tests (unit, integration, e2e, I/O behavioural) | ~1h |
| Memory integration + demo + README | ~0.45h |
| **Total** | **~4.45h** |
