"""
TEST_PLAN 1.2 — Static constraint validation (no LLM, instant).
TEST_PLAN 5.2 — README and demo artifact checklist.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent.parent

BANNED_PACKAGES = [
    "langchain", "llamaindex", "llama_index", "llama-index",
    "langgraph", "crewai", "autogen", "semantic-kernel",
    "haystack", "mem0", "letta", "memgpt", "zep", "motorhead",
]


# ── 1.2 Banned framework check ───────────────────────────────────────────────

def test_no_banned_frameworks_in_requirements():
    """requirements.txt must not contain any banned agent/memory frameworks."""
    req = (ROOT / "requirements.txt").read_text().lower()
    for pkg in BANNED_PACKAGES:
        assert pkg not in req, (
            f"Banned package '{pkg}' found in requirements.txt"
        )


def test_no_banned_frameworks_imported_in_src():
    """Source files must not import any banned framework."""
    src_files = list((ROOT / "src").rglob("*.py"))
    for path in src_files:
        text = path.read_text().lower()
        for pkg in BANNED_PACKAGES:
            # match `import pkg` or `from pkg`
            assert not re.search(rf"\b(import|from)\s+{re.escape(pkg)}", text), (
                f"Banned import '{pkg}' found in {path.relative_to(ROOT)}"
            )


# ── 5.2 README and demo artifact checklist ───────────────────────────────────

def test_readme_exists():
    assert (ROOT / "README.md").exists(), "README.md is missing"


def test_readme_contains_design_section():
    text = (ROOT / "README.md").read_text()
    assert re.search(r"[Dd]esign", text), "README missing design section"


def test_readme_contains_tradeoffs_section():
    text = (ROOT / "README.md").read_text()
    assert re.search(r"[Tt]radeoff", text), "README missing tradeoffs section"


def test_readme_contains_time_spent():
    text = (ROOT / "README.md").read_text()
    assert re.search(r"[Tt]ime\s+[Ss]pent", text), "README missing 'Time Spent' section"


def test_readme_contains_next_steps():
    text = (ROOT / "README.md").read_text()
    assert re.search(r"[Nn]ext|[Ff]uture|[Bb]uild next", text), (
        "README missing what-to-build-next section"
    )


def test_demo_artifact_exists():
    """A runnable demo script must exist at the project root."""
    assert (ROOT / "demo.py").exists(), "demo.py is missing"


def test_env_example_exists():
    assert (ROOT / ".env.example").exists(), ".env.example is missing"


def test_requirements_file_exists():
    assert (ROOT / "requirements.txt").exists(), "requirements.txt is missing"
