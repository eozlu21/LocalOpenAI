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

## Configuration

### Changing Models
Edit `app/model.py` and modify the `model_name` parameter:
```python
def __init__(self, model_name="your-preferred-model"):
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