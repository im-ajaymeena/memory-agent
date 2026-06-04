from dataclasses import dataclass


@dataclass
class Memory:
    id: str
    body: str
    type: str                 # e.g. "preferences_interests"
    source: str               # "user_statement" | "agent_inference"
    updated_at: str           # ISO 8601
    age_human_readable: str   # e.g. "3 days ago"


class VanillaMemory:
    """No-op memory implementation used in tests to isolate the agent pipeline."""

    def __init__(self) -> None:
        self._store: list[Memory] = []

    def retrieve(self, query: str) -> list[Memory]:
        return self._store[:5]

    async def extract_and_store(self, turns: list) -> None:
        pass
