"""
Stress test for finding a small signal in a massive amount of conversational noise.
"""
import asyncio
import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

from src.agent import chat
from src.extractor import drain_pending_extraction, init_extractor, set_memory
from src.memory.real import RealMemory
from src.session import Session

async def main():
    print("--- Initialising RealMemory for Large Noise Stress Test ---")
    db_path = pathlib.Path("~/.agent/memories/memories_large_noise.db").expanduser()
    if db_path.exists():
        os.remove(db_path)
    
    mem = RealMemory(db_path=db_path)
    set_memory(mem)
    init_extractor()
    s = Session()
    
    # -------------------------------------------------------------------------
    # The Payload: 90% Noise, 10% Critical Signal
    # Signal: "I am severely allergic to penicillin."
    # Signal: "My emergency contact is my brother, Liam."
    # Noise: A rambly story about going to the grocery store and making banana bread.
    # -------------------------------------------------------------------------
    massive_query = """
    Hey there, I just wanted to tell you about my day. It started off raining really heavily, 
    which is super annoying because I had to walk to the grocery store. I needed to get bananas, 
    flour, sugar, and eggs because I was planning to make this amazing banana bread recipe that 
    my grandmother gave me. Speaking of my grandmother, she used to live in Ohio but moved to 
    Florida recently. Anyway, the grocery store was packed! I couldn't find the right kind of 
    flour, they only had whole wheat and I really needed all-purpose. While I was in the aisle, 
    I ran into an old friend from high school, which was crazy. We talked for like twenty minutes 
    about our old teachers. 

    Oh, by the way, this is super important for you to remember just in case: I am severely 
    allergic to penicillin. 

    So anyway, after the store, I walked back in the rain, got completely soaked, and realized 
    I forgot the eggs! Can you believe it? I had to go all the way back. By the time I got home, 
    I was exhausted. I didn't even end up making the banana bread, I just ordered a pizza instead. 
    The pizza was pretty good though, it had pepperoni and jalapeños. 

    Also, another quick thing for my profile, my emergency contact is my brother, Liam. 

    I'm going to try making the banana bread tomorrow instead, assuming the weather is better. 
    Do you have any tips for making banana bread super moist?
    """
    
    print("\n--- Sending Massive Noisy Query ---")
    print("Query Length:", len(massive_query), "characters")
    print("Extracting...")
    
    response, _ = await chat(massive_query, s)
    
    print(f"\nAgent Response:\n{response}")
        
    print("\n[Waiting for async extraction to finish...]")
    await drain_pending_extraction(timeout_s=30.0)
    
    print(f"\n{'='*60}")
    print(f"--- Final Memory State (What survived the noise?) ---")
    print(f"{'='*60}")
    memories = mem.all()
    if not memories:
        print("FAIL: The agent extracted absolutely nothing (Signal lost).")
    for m in memories:
        print(f"[{m.type}] {m.body}")

if __name__ == "__main__":
    asyncio.run(main())
