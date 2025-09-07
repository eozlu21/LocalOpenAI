import asyncio
from openai import AsyncOpenAI

# === Configuration ===
API_BASE = "http://ai18.kuvalar.ku.edu.tr:8000/v1"
MODEL    = "microsoft/Phi-4-mini-instruct"

# === Instantiate async client ===
client = AsyncOpenAI(
    api_key="not-needed",    # ignored by local server
    base_url=API_BASE,       # points to your FastAPI endpoint
)

async def make_request(user_message="Tell me a joke.", request_id=1):
    """Async function to make the API request"""
    print(f"[Request {request_id}] Starting request...")
    
    # === Send request ===
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.7,
        max_tokens=256,
        stop=["\n[User]:"]
    )
    
    # === Print the answer ===
    print(f"[Request {request_id}] Response: {response.choices[0].message.content}")
    print(f"[Request {request_id}] Completed!")
    return response

async def make_parallel_requests():
    """Make multiple concurrent requests to test parallel processing"""
    print("=== Making parallel requests ===")
    
    # Different prompts for each request
    prompts = [
        "Tell me a joke.",
        "What's the weather like?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about programming.",
        "What's the capital of France?"
    ]
    
    # Create tasks for concurrent execution
    tasks = [
        make_request(prompt, i+1) 
        for i, prompt in enumerate(prompts)
    ]
    
    # Measure time
    import time
    start_time = time.time()
    
    # Execute all requests concurrently
    responses = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"\n=== All {len(responses)} requests completed in {end_time - start_time:.2f} seconds ===")
    return responses

async def make_sequential_requests():
    """Make requests sequentially for comparison"""
    print("=== Making sequential requests ===")
    
    prompts = [
        "Tell me a joke.",
        "What's the weather like?",
        "Explain quantum computing in simple terms."
    ]
    
    import time
    start_time = time.time()
    
    responses = []
    for i, prompt in enumerate(prompts):
        response = await make_request(prompt, i+1)
        responses.append(response)
    
    end_time = time.time()
    
    print(f"\n=== All {len(responses)} sequential requests completed in {end_time - start_time:.2f} seconds ===")
    return responses

async def main():
    """Main async function"""
    print("=== Testing Different Request Patterns ===\n")
    
    # Test 1: Single request
    print("1. Single request test:")
    response = await make_request()
    print(f"   Model used: {response.model}\n")
    
    # Test 2: Sequential requests
    print("2. Sequential requests test:")
    await make_sequential_requests()
    print()
    
    # Test 3: Parallel requests (the main benefit of async!)
    print("3. Parallel requests test:")
    await make_parallel_requests()
    print()
    
    print("=== All tests completed! ===")

# === Run the async code ===
if __name__ == "__main__":
    asyncio.run(main())
