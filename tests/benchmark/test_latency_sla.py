"""
Latency SLA test: p50 TTFT at 1K memories must be within 200ms of cold p50.

Requires a real ANTHROPIC_API_KEY. Marked @pytest.mark.slow.
Run with: make test-benchmark
"""
import asyncio
import json
import subprocess
import sys

import pytest


pytestmark = pytest.mark.slow


@pytest.mark.asyncio
async def test_latency_sla_passes():
    """Run benchmark.py and assert the SLA from the PRD is met."""
    result = subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--output-json"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"

    data = json.loads(result.stdout.strip().splitlines()[-1])
    delta = data["delta_ms"]
    assert data["sla_pass"], (
        f"Latency SLA violated: p50 delta = {delta}ms (limit: 200ms)\n"
        f"  cold={data['p50_cold_ms']}ms  warm={data['p50_warm_ms']}ms"
    )
