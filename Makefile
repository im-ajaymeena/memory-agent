.PHONY: install test test-e2e test-benchmark benchmark demo run

install:
	pip install -r requirements.txt

# Fast: unit + integration only (mocked LLM, no API key needed for these)
test:
	pytest tests/unit tests/integration -q

# All fast tests with coverage report
test-cov:
	pytest tests/unit tests/integration --tb=short -v

# E2E: real LLM calls — requires ANTHROPIC_API_KEY
test-e2e:
	pytest tests/e2e -m slow -v

# Latency SLA — runs 60 real API calls, takes ~5 min
test-benchmark:
	pytest tests/benchmark -m slow -v

# Run the interactive CLI
run:
	python -m src.cli

# Run the CLI and resume a session (usage: make run SESSION=<id>)
run-session:
	python -m src.cli --session-id $(SESSION)

# List all saved sessions
sessions:
	python -m src.cli --list-sessions

# Latency benchmark standalone
benchmark:
	python testing-scripts/benchmark.py

# Cross-session memory demo
demo:
	python demo.py
