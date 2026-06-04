from typing import Protocol, runtime_checkable
from .stub import Memory


@runtime_checkable
class MemoryInterface(Protocol):
    def retrieve(self, query: str) -> list[Memory]: ...
    async def extract_and_store(self, turns: list) -> None: ...
