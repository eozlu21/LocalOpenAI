# LocalOpenAI - OpenAI-Compatible Local API Server

A lightweight FastAPI server that exposes local language models (Phi-4, Qwen, etc.) through OpenAI's ChatCompletion API interface. Built with vLLM for efficient inference.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint
- **Local inference**: Run models entirely on your hardware
- **Multi-user support**: Handle concurrent requests
- **Flexible models**: Support for Hugging Face transformers (default: Phi-4-mini-instruct)

## Quick Start

### Server Setup

1. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Start the server**:

    ```bash
    ./start_server.sh
    ```

The server will run on `http://localhost:8000` (or `0.0.0.0:8000` for network access).

### Client Usage

#### Option 1: OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="microsoft/Phi-4-mini-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing briefly."}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)
```

#### Option 2: cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-4-mini-instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## Updated Client Usage

### Async Python Client

The client now supports asynchronous requests for better performance and scalability. Here's how to use it:

```python
import asyncio
from openai import AsyncOpenAI

# === Configuration ===
API_BASE = "http://localhost:8000/v1"
MODEL    = "microsoft/Phi-4-mini-instruct"

# === Instantiate async client ===
client = AsyncOpenAI(
    api_key="not-needed",    # ignored by local server
    base_url=API_BASE,       # points to your FastAPI endpoint
)

async def make_request():
    """Async function to make the API request"""
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
    
    print(response.choices[0].message.content)
    return response

async def main():
    response = await make_request()
    print(f"Request completed. Model used: {response.model}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Parallel Requests

You can also make multiple concurrent requests to test parallel processing:

```python
async def make_parallel_requests():
    prompts = [
        "Tell me a joke.",
        "What's the weather like?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about programming.",
        "What's the capital of France?"
    ]

    tasks = [make_request(prompt, i+1) for i, prompt in enumerate(prompts)]
    responses = await asyncio.gather(*tasks)

    for response in responses:
        print(response.choices[0].message.content)
```

This demonstrates the power of async operations for handling multiple requests efficiently.

## Configuration

### Changing Models

Edit `app/model.py` and modify the `model_name` parameter:

```python
# Example
class QwenLocalModel:
    def __init__(self, model_name="your-preferred-model"):
        ...existing code...
```

### API Parameters

- `temperature`: Controls randomness (0.0-2.0)
- `max_tokens`: Maximum response length
- `stop`: Stop sequences to end generation
- `model`: Model identifier (informational)

## Project Structure

```
LocalOpenAI/
├── app/
│   ├── main.py           # FastAPI application
│   ├── openai_api.py     # OpenAI-compatible endpoints
│   └── model.py          # Model loading and inference
├── example/
│   └── client.py         # Example client usage
├── requirements.txt      # Dependencies
├── start_server.sh       # Server launcher
└── README.md
```

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM for Phi-4-mini-instruct

## Troubleshooting

- **Port conflicts**: Change port in `start_server.sh`
- **Memory issues**: Try smaller models or reduce `max_tokens`
- **Connection refused**: Ensure server is running and accessible

## License

Open source - contributions welcome!