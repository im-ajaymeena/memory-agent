import json
import pathlib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

SESSIONS_DIR = pathlib.Path("~/.agent/sessions").expanduser()


@dataclass
class Turn:
    role: str       # "user" | "assistant"
    content: str
    timestamp: str  # ISO 8601
    id: str = ""    # uuid — used as extraction cursor (see extractor.py)

    @staticmethod
    def now(role: str, content: str) -> "Turn":
        ts = datetime.now(timezone.utc).isoformat()
        return Turn(role=role, content=content, timestamp=ts, id=str(uuid.uuid4()))


class Session:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.path = SESSIONS_DIR / f"{self.session_id}.jsonl"
        self.history: list[Turn] = []
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> "Session":
        """Replay all events from disk to reconstruct history."""
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.history.append(Turn(**json.loads(line)))
        return self

    def append(self, turn: Turn) -> None:
        self.history.append(turn)
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(turn)) + "\n")
            f.flush()  # defeats OS buffering: crash-safe

    def last_n_turns(self, n: int) -> list[Turn]:
        return self.history[-n:]

    @staticmethod
    def list_all() -> list[dict]:
        if not SESSIONS_DIR.exists():
            return []
        return [
            {"id": p.stem, "path": str(p), "mtime": p.stat().st_mtime}
            for p in sorted(
                SESSIONS_DIR.glob("*.jsonl"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
        ]
