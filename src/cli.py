import argparse
import asyncio
import json
import pathlib
import sys
import textwrap

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

load_dotenv()

from .agent import chat
from .extractor import drain_pending_extraction, init_extractor
from .session import Session, Turn

HISTORY_FILE = pathlib.Path("~/.agent/repl_history").expanduser()
REPLAY_TURNS = 5          # how many prior turns to show on resume
TITLE_MAX_LEN = 60        # characters for session title in /sessions list
WRAP_WIDTH = 80           # assistant reply wrap width in replay


# ── session title ─────────────────────────────────────────────────────────────

def _session_title(path: str) -> str:
    """
    Peek at the JSONL file and return the first user message as a title.
    Reads only the minimum number of lines — no full session load.
    """
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turn = json.loads(line)
                if turn.get("role") == "user":
                    content = turn.get("content", "").replace("\n", " ").strip()
                    if len(content) > TITLE_MAX_LEN:
                        return content[:TITLE_MAX_LEN] + "…"
                    return content
    except Exception:
        pass
    return "(empty)"


# ── conversation replay ───────────────────────────────────────────────────────

def _print_replay(session: Session, n: int = REPLAY_TURNS) -> None:
    """Print the last n turns as if the conversation were already running."""
    turns = session.last_n_turns(n)
    if not turns:
        return

    skipped = len(session.history) - len(turns)
    print(f"\n{'─'*60}")
    if skipped > 0:
        print(f"  ··· {skipped} earlier turn{'s' if skipped > 1 else ''} ···")

    for turn in turns:
        if turn.role == "user":
            print(f"\n\033[1m> {turn.content}\033[0m")
        else:
            print()
            # Wrap long assistant replies so they don't scroll off-screen
            for line in turn.content.splitlines():
                if line:
                    for wrapped in textwrap.wrap(line, WRAP_WIDTH) or [line]:
                        print(wrapped)
                else:
                    print()

    print(f"{'─'*60}\n")


# ── repl ──────────────────────────────────────────────────────────────────────

async def repl(session_id: str | None) -> None:
    init_extractor()

    session = Session(session_id).load()
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    prompt = PromptSession(history=FileHistory(str(HISTORY_FILE)))
    abort = asyncio.Event()

    # On startup: if resuming an existing session, replay the last turns
    if session.history:
        title = _session_title(str(session.path))
        print(f"Session: {session.session_id[:8]}...  |  {len(session.history)} turns  |  \"{title}\"")
        print("Commands: /quit  /sessions  /clear  /resume [# or id]  /memories  /forget <id>  (ctrl-C aborts)\n")
        _print_replay(session)
    else:
        print(f"Session: {session.session_id[:8]}...  |  New session")
        print("Commands: /quit  /sessions  /clear  /resume [# or id]  /memories  /forget <id>  (ctrl-C aborts)\n")

    while True:
        try:
            user_input = await prompt.prompt_async("> ")
        except KeyboardInterrupt:
            abort.set()
            await asyncio.sleep(0.05)
            abort.clear()
            print()
            continue
        except EOFError:
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input == "/quit":
            break

        if user_input == "/memories":
            from .agent import memory
            memories = memory.all()
            if not memories:
                print("  No memories stored yet.")
            else:
                print(f"\n  {len(memories)} stored memories:\n")
                for i, m in enumerate(memories, 1):
                    print(f"  [{i}] {m.id[:8]}  [{m.type}]  {m.body}")
                    print(f"        source: {m.source}  |  {m.age_human_readable}")
            print()
            continue

        if user_input.startswith("/forget"):
            arg = user_input[len("/forget"):].strip()
            if not arg:
                print("  Usage: /forget <memory-id-prefix>")
                continue
            from .agent import memory
            all_memories = memory.all()
            matches = [m for m in all_memories if m.id.startswith(arg)]
            if not matches:
                print(f"  No memory matching '{arg}'. Use /memories to list IDs.")
            elif len(matches) > 1:
                print(f"  Ambiguous — {len(matches)} memories match '{arg}'. Be more specific.")
            else:
                m = matches[0]
                memory._store.soft_delete(m.id)
                print(f"  Forgot: [{m.type}] {m.body}")
            continue

        if user_input == "/sessions":
            sessions = Session.list_all()[:10]
            if not sessions:
                print("  No sessions found.")
            for i, s in enumerate(sessions, 1):
                title = _session_title(s["path"])
                print(f"  [{i}] {s['id'][:8]}...  \"{title}\"")
            continue

        if user_input.startswith("/resume"):
            arg = user_input[len("/resume"):].strip()
            sessions = Session.list_all()
            target_id = None

            if arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(sessions):
                    target_id = sessions[idx]["id"]
                else:
                    print(f"  No session at index {arg}.")
            elif arg:
                matches = [s for s in sessions if s["id"].startswith(arg)]
                if len(matches) == 1:
                    target_id = matches[0]["id"]
                elif len(matches) == 0:
                    print(f"  No session matching '{arg}'.")
                else:
                    print(f"  Ambiguous — {len(matches)} sessions match '{arg}'.")
            else:
                # No arg: show list with titles, prompt for pick
                if not sessions:
                    print("  No sessions found.")
                    continue
                for i, s in enumerate(sessions[:10], 1):
                    title = _session_title(s["path"])
                    print(f"  [{i}] {s['id'][:8]}...  \"{title}\"")
                try:
                    pick = await prompt.prompt_async("  Resume [#]: ")
                    pick = pick.strip()
                    if pick.isdigit():
                        idx = int(pick) - 1
                        if 0 <= idx < len(sessions[:10]):
                            target_id = sessions[idx]["id"]
                        else:
                            print("  Invalid number.")
                    else:
                        print("  Cancelled.")
                except (KeyboardInterrupt, EOFError):
                    print()
                    continue

            if target_id:
                await drain_pending_extraction(timeout_s=10.0)
                session = Session(target_id).load()
                init_extractor()
                title = _session_title(str(session.path))
                print(f"\nResumed: {session.session_id[:8]}...  |  {len(session.history)} turns  |  \"{title}\"")
                _print_replay(session)
            continue

        if user_input == "/clear":
            await drain_pending_extraction(timeout_s=10.0)
            session = Session()
            init_extractor()
            print(f"New session: {session.session_id[:8]}...\n")
            continue

        print()
        try:
            _, ttft = await chat(user_input, session, abort=abort)
            print(f"\n[ttft: {ttft * 1000:.0f}ms]\n")
        except Exception as e:
            print(f"\n[error] {e}\n")

    print("\n[flushing memory extraction...]", end="", flush=True)
    try:
        await drain_pending_extraction(timeout_s=30.0)
        print(" done.")
    except asyncio.TimeoutError:
        print(" timed out.")
    print("Goodbye.")


def main() -> None:
    args = parse_args()
    if args.list_sessions:
        for s in Session.list_all():
            title = _session_title(s["path"])
            print(f"{s['id']}  \"{title}\"")
        sys.exit(0)
    asyncio.run(repl(args.session_id))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conversational agent with persistent memory")
    p.add_argument("--session-id", help="Resume a prior session by ID")
    p.add_argument("--list-sessions", action="store_true", help="List recent sessions and exit")
    return p.parse_args()


if __name__ == "__main__":
    main()
