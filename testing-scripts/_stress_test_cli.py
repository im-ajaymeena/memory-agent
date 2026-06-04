"""
Stress test for complex memory updates, contradictions, and rapid-fire turns across sessions.
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from src.agent import chat
from src.extractor import drain_pending_extraction, init_extractor, set_memory
from src.memory.real import RealMemory
from src.session import Session

async def run_session(session_num, mem, turns):
    print(f"\n{'='*60}")
    print(f"--- Session {session_num} ---")
    print(f"{'='*60}")
    init_extractor()
    s = Session()
    
    for user_msg in turns:
        print(f"\nUser: {user_msg}")
        response, _ = await chat(user_msg, s)
        print(f"Agent: {response}")
        
    print("\n[Waiting for extraction to finish...]")
    await drain_pending_extraction(timeout_s=20.0)
    
    print(f"\n--- Session {session_num} Memory State ---")
    memories = mem.all()
    for m in memories:
        print(f"[{m.type}] {m.body} (Source: {m.source})")

async def main():
    print("--- Initialising RealMemory for Stress Test ---")
    mem = RealMemory() # Uses default db or custom if we pass one, we'll use default which might have previous data. 
    # Let's clear the default db for a clean test
    import pathlib, os
    db_path = pathlib.Path("~/.agent/memories/memories.db").expanduser()
    if db_path.exists():
        os.remove(db_path)
    
    mem = RealMemory()
    set_memory(mem)
    
    # Session 1: Establish baseline facts
    turns1 = [
        "I live in New York and I absolutely hate coffee.",
        "Also, my dog is named Rex."
    ]
    await run_session(1, mem, turns1)
    
    # Session 2: Contradictions and Corrections
    turns2 = [
        "Actually, I just moved to London last week.",
        "Wait, I misspoke earlier. I love coffee, it's tea that I hate.",
        "What's my pet's name?"
    ]
    await run_session(2, mem, turns2)
    
    # Session 3: Deep Correction & Recall
    turns3 = [
        "I don't have a dog, I was joking. Rex is actually an iguana.",
        "Can you summarize everything you know about me?"
    ]
    await run_session(3, mem, turns3)
    
    # Session 4: Rapid-fire turns (Queue/Coalescing Stress Test)
    print(f"\n{'='*60}")
    print("--- Session 4: Rapid Fire Stress Test ---")
    print(f"{'='*60}")
    init_extractor()
    s4 = Session()
    rapid_turns = [
        "One.",
        "My favorite color is blue.",
        "Two.",
        "Three.",
        "Actually, my favorite color is red."
    ]
    
    # Don't wait between these, blast them into the chat loop
    for user_msg in rapid_turns:
        print(f"User: {user_msg}")
        # Note: chat awaits the LLM response, but extraction is async background
        response, _ = await chat(user_msg, s4)
        
    print("\n[Waiting for rapid-fire extraction to finish...]")
    await drain_pending_extraction(timeout_s=30.0)
    
    print(f"\n--- Final Memory State ---")
    memories = mem.all()
    for m in memories:
        print(f"[{m.type}] {m.body} (Source: {m.source})")

if __name__ == "__main__":
    asyncio.run(main())
