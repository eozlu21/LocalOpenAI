# LocalOpenAI - OpenAI-Compatible Local API Server

A lightweight FastAPI server that exposes local language models (Qwen, Phi-4, etc.) through OpenAI's ChatCompletion API interface. Built with vLLM for efficient inference.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint
- **Local inference**: Run models entirely on your hardware
- **Multi-user support**: Handle concurrent requests
- **Flexible models**: Support for Hugging Face transformers (default: Qwen/Qwen2.5-32B-Instruct)

## Quick Start

### Server Setup

1. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Start the server (basic)**:
```bash
./start_server.sh
```

The server will run on `http://localhost:8001` (or `0.0.0.0:8001`).

3. **Start with custom model / GPU count / port**:
```bash
./start_server.sh \
  --model Qwen/Qwen2.5-32B-Instruct \
  --gpus 2 \
  --port 9000
```
Flags:
- `--model` Hugging Face model id (sets `LOCAL_OPENAI_MODEL`)
- `--gpus` Tensor parallel size (sets `LOCAL_OPENAI_TP`)
- `--port` HTTP port (default 8001)

Environment variable equivalents (optional):
```bash
export LOCAL_OPENAI_MODEL=Qwen/Qwen2.5-32B-Instruct
export LOCAL_OPENAI_TP=2
./start_server.sh
```

### Client Usage

#### Option 1: OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
  api_key="not-needed",
  base_url="http://localhost:8001/v1"
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
curl http://localhost:8001/v1/chat/completions \
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

### Changing Models & GPU Count
Prefer the CLI:
```bash
./start_server.sh --model mistralai/Mistral-7B-Instruct-v0.3 --gpus 4
```
Or environment variables:
```bash
export LOCAL_OPENAI_MODEL=mistralai/Mistral-7B-Instruct-v0.3
export LOCAL_OPENAI_TP=4
./start_server.sh
```
Notes:
- The `model` field you send in the request body is informational; the actually loaded model is set at startup.
- `--gpus/LOCAL_OPENAI_TP` configures vLLM `tensor_parallel_size` (must not exceed visible CUDA devices).
- If omitted, tensor parallel is only set if you passed `--gpus`; otherwise vLLM defaults to single GPU.

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
- Adequate GPU memory for chosen model (e.g., 7B models typically require ≥14–16GB total across GPUs)

## Troubleshooting

- **Port conflicts**: Use `--port` flag
- **Memory issues**: Try smaller models or reduce `max_tokens`
- **Connection refused**: Ensure server is running and accessible
 - **Wrong model in responses**: Confirm you started server with `--model`; request body `model` does not trigger reloading.

## License

Open source - contributions welcome!