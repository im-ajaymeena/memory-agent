"""
Inject N synthetic memories directly into the SQLite store.
Used to pre-populate a known-size store for latency benchmarking.

Usage:
    python scripts/populate_store.py 1000
    python scripts/populate_store.py 1000 --db /tmp/bench.db
"""
import argparse
import sys
import pathlib

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.memory.embedder import embed_batch
from src.memory.models import Category, MemoryRecord, Source
from src.memory.store import MemoryStore

TEMPLATES: list[tuple[Category, str]] = [
    (Category.PREFERENCES,   "User prefers {lang} for scripting tasks."),
    (Category.PROFESSIONAL,  "User is a {role} at a {domain} company."),
    (Category.PERSONAL,      "User's name is {name}. Based in {city}."),
    (Category.GOALS,         "User is learning {skill} to improve {goal}."),
    (Category.CONTEXTUAL,    "User is currently working on {project}."),
]
FILLS: dict[str, list[str]] = {
    "lang":    ["Python", "Go", "TypeScript", "Rust", "Kotlin"],
    "role":    ["backend engineer", "senior SWE", "tech lead", "staff engineer", "SRE"],
    "domain":  ["fintech", "healthtech", "SaaS", "infrastructure", "security"],
    "name":    ["Alex", "Sam", "Jordan", "Taylor", "Morgan"],
    "city":    ["San Francisco", "New York", "London", "Berlin", "Singapore"],
    "skill":   ["Rust", "Kubernetes", "ML infra", "systems design", "distributed systems"],
    "goal":    ["career growth", "system performance", "team leadership", "side projects"],
    "project": ["auth service refactor", "payment pipeline", "API gateway", "data ingestion"],
}


def populate(store: MemoryStore, n: int) -> None:
    texts = []
    metas = []
    for i in range(n):
        cat, tpl = TEMPLATES[i % len(TEMPLATES)]
        body = tpl.format(**{k: v[i % len(v)] for k, v in FILLS.items()})
        texts.append(body)
        metas.append(cat)

    print(f"Embedding {n} texts...", flush=True)
    embeddings = embed_batch(texts)

    for i, (text, cat, emb) in enumerate(zip(texts, metas, embeddings)):
        store.add(MemoryRecord(
            text=text,
            category=cat,
            source=Source.USER_STATEMENT,
            embedding=emb.tolist(),
        ))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} inserted", flush=True)

    print(f"Done. Store now has {store.count_active()} active memories.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int, help="Number of synthetic memories to inject")
    p.add_argument("--db", default=None, help="Path to SQLite db (default: ~/.agent/memories/memories.db)")
    args = p.parse_args()

    db_path = pathlib.Path(args.db).expanduser() if args.db else None
    from src.memory.store import MemoryStore
    from src.memory.real import _DEFAULT_DB
    store = MemoryStore(db_path or _DEFAULT_DB)
    populate(store, args.n)
