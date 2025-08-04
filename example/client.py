import asyncio
from openai import AsyncOpenAI

# === Configuration ===
API_BASE = "http://ai21.kuvalar.ku.edu.tr:8000/v1"
MODEL    = "microsoft/Phi-4-mini-instruct"

# === Instantiate async client ===
client = AsyncOpenAI(
    api_key="not-needed",    # ignored by local server
    base_url=API_BASE,       # points to your FastAPI endpoint
)

async def make_request():
    """Async function to make the API request"""
    # === Send request ===
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Tell me a joke."}
        ],
        temperature=0.7,
        max_tokens=256,
        stop=["\n[User]:"]
    )
    
    # === Print the answer ===
    print(response.choices[0].message.content)
    return response

async def main():
    """Main async function"""
    print("Making async request...")
    response = await make_request()
    print(f"Request completed. Model used: {response.model}")

# === Run the async code ===
if __name__ == "__main__":
    asyncio.run(main())
