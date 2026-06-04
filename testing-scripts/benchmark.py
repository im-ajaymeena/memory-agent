"""
Measures p50 TTFT at cold store (0 memories) vs warm store (1K memories).
Reports delta and asserts < 200ms PRD SLA.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --output-json
    python scripts/benchmark.py --n-runs 10   # quicker smoke run
"""
import argparse
import asyncio
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import anthropic
from src.memory.real import RealMemory
from src.memory.store import MemoryStore
from scripts.populate_store import populate

N_RUNS_DEFAULT = 30
TEST_QUERIES = [
    "What language do I prefer?",
    "What project am I working on?",
    "What's my job title?",
    "Tell me about myself.",
    "What are my current goals?",
]

_client = anthropic.AsyncAnthropic()


def _build_system(mem: RealMemory, query: str) -> str:
    base = "You are a helpful assistant. You remember facts about this user."
    memories = mem.retrieve(query)
    if not memories:
        return base
    facts = "\n".join(f"- [{m.type} | {m.age_human_readable}] {m.body}" for m in memories)
    return f"{base}\n\nKnown facts:\n{facts}"


async def _single_ttft(system: str, query: str) -> float:
    t_send = time.perf_counter()
    ttft = 0.0
    async with _client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=64,
        system=system,
        messages=[{"role": "user", "content": query}],
    ) as stream:
        async for _ in stream.text_stream:
            ttft = time.perf_counter() - t_send
            break
    return ttft


async def measure_ttfts(label: str, mem: RealMemory, n_runs: int) -> list[float]:
    ttfts: list[float] = []
    for i in range(n_runs):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        system = _build_system(mem, query)
        ttft = await _single_ttft(system, query)
        ttfts.append(ttft)
        print(f"  [{label}] run {i+1:2d}/{n_runs}: {ttft*1000:.0f}ms", flush=True)
    return ttfts


async def main(output_json: bool = False, n_runs: int = N_RUNS_DEFAULT) -> None:
    print("=== Latency Benchmark ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        cold_db = pathlib.Path(tmpdir) / "cold.db"
        warm_db = pathlib.Path(tmpdir) / "warm.db"

        cold_mem = RealMemory(db_path=cold_db)
        print(f"COLD store (0 memories)")
        cold = await measure_ttfts("COLD", cold_mem, n_runs)

        warm_mem = RealMemory(db_path=warm_db)
        populate(warm_mem._store, 1000)
        print(f"\n1K-MEMORY store ({warm_mem.count()} memories)")
        warm = await measure_ttfts("1K  ", warm_mem, n_runs)

    p50_cold = float(np.percentile(cold, 50)) * 1000
    p50_warm = float(np.percentile(warm, 50)) * 1000
    delta = p50_warm - p50_cold

    result = {
        "p50_cold_ms": round(p50_cold, 1),
        "p50_warm_ms": round(p50_warm, 1),
        "delta_ms":    round(delta, 1),
        "sla_pass":    delta < 200,
        "n_runs":      n_runs,
    }

    if output_json:
        print(json.dumps(result))
    else:
        print(f"\n{'='*40}")
        print(f"p50 COLD     : {result['p50_cold_ms']}ms")
        print(f"p50 1K-MEM   : {result['p50_warm_ms']}ms")
        print(f"Delta        : {result['delta_ms']}ms")
        print(f"SLA (<200ms) : {'PASS' if result['sla_pass'] else 'FAIL'}")
        print(f"{'='*40}")

    if not result["sla_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", action="store_true")
    p.add_argument("--n-runs", type=int, default=N_RUNS_DEFAULT)
    args = p.parse_args()
    asyncio.run(main(args.output_json, args.n_runs))
