import json
import re

import anthropic
import numpy as np

from .embedder import embed
from .models import CandidateFact, MemoryRecord, SOURCE_TRUST
from .store import MemoryStore

_client = anthropic.AsyncAnthropic()

_PROMPT = """\
You are a memory consistency arbiter. Given a NEW CANDIDATE fact and the most semantically similar EXISTING MEMORIES, decide the single correct operation.

Operations:
  ADD    — candidate is genuinely new; no existing memory covers it
  UPDATE — candidate enriches or partially corrects an existing memory (provide target_id)
  DELETE — candidate EXPLICITLY supersedes an existing memory; the old record is NOW OUTDATED AND WRONG (provide target_id of the OLD record)
  NONE   — candidate is already accurately captured by an existing memory

Rules:
  - user_statement beats agent_inference for the same content → prefer NONE or UPDATE to preserve user's version
  - When unsure between ADD and NONE → prefer NONE (avoid duplicates)
  - When unsure between UPDATE and DELETE → prefer UPDATE (less destructive)
  - Use DELETE when the candidate uses language like "moved from X to Y", "switched from X to Y", "left X", "no longer at X":
      existing: "User lives in New York"  +  candidate: "User moved from New York to San Francisco"  → DELETE the New York record
      existing: "User uses Python"        +  candidate: "User switched from Python to Rust"          → DELETE the Python record
      existing: "User works at Acme"      +  candidate: "User left Acme, now at Google"              → DELETE the Acme record

Return ONLY valid JSON, no prose, no code fences.
Example outputs:
  {"operation": "ADD", "target_id": null}
  {"operation": "UPDATE", "target_id": "<id>"}
  {"operation": "DELETE", "target_id": "<id of the OLD record to remove>"}
  {"operation": "NONE", "target_id": null}
"""

_PROMPT_SUFFIX = """
NEW CANDIDATE:
{candidate}

EXISTING MEMORIES (most similar first):
{existing}
"""


async def adjudicate(candidate: CandidateFact, store: MemoryStore) -> None:
    candidate_vec = embed(candidate.text)
    similar = store.search(candidate_vec, k=10)

    # Deterministic fast-path: existing record with same-or-higher trust + near-identical content → NONE
    if similar:
        top = similar[0]
        top_vec = np.array(top.embedding, dtype=np.float32)
        cosine = float(candidate_vec @ top_vec)
        if top.source_trust >= SOURCE_TRUST[candidate.source] and cosine > 0.92:
            return  # already captured — skip LLM

    existing_json = json.dumps(
        [
            {
                "id": r.id,
                "text": r.text,
                "source": r.source.value,
                "trust": r.source_trust,
                "updated": r.timestamp_updated,
            }
            for r in similar
        ],
        indent=2,
    )
    candidate_json = json.dumps({
        "text": candidate.text,
        "category": candidate.category.value,
        "source": candidate.source.value,
        "intent_label": candidate.intent_label,
    })

    response = await _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": _PROMPT + _PROMPT_SUFFIX.format(
                candidate=candidate_json,
                existing=existing_json,
            ),
        }],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences the model sometimes adds
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        result = json.loads(match.group() if match else raw)
    except (json.JSONDecodeError, AttributeError):
        result = {"operation": "ADD", "target_id": None}

    op: str = result.get("operation", "NONE")
    target_id: str | None = result.get("target_id")

    if op == "ADD":
        store.add(MemoryRecord(
            text=candidate.text,
            category=candidate.category,
            source=candidate.source,
            intent_label=candidate.intent_label,
            entities=candidate.entities,
            contextual_markers=candidate.contextual_markers,
            embedding=candidate_vec.tolist(),
        ))

    elif op == "UPDATE" and target_id:
        store.update(target_id, candidate.text, candidate_vec.tolist())

    elif op == "DELETE" and target_id:
        store.soft_delete(target_id)
        store.add(MemoryRecord(
            text=candidate.text,
            category=candidate.category,
            source=candidate.source,
            intent_label=candidate.intent_label,
            entities=candidate.entities,
            contextual_markers=candidate.contextual_markers,
            embedding=candidate_vec.tolist(),
        ))
    # NONE: no-op
