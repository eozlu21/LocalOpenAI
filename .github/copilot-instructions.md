# Copilot Instructions for LocalOpenAI

Purpose: Help AI coding agents contribute quickly to this FastAPI + vLLM server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

## Architecture at a glance
- `app/main.py`: FastAPI app bootstrap; includes the OpenAI-compatible router.
- `app/openai_api.py`: HTTP layer. Validates payloads (model, messages, temperature, max_tokens, stop), builds a prompt, and calls the model. Logs every request and message to terminal.
- `app/model.py`: Inference layer using vLLM. Loads the model (default `microsoft/Phi-4-mini-instruct`), creates `SamplingParams`, and exposes `generate()` and `generate_async()`.
- `example/client.py`: Async demo using OpenAI Python SDK pointed at this server; includes single, sequential, and parallel request patterns.
- `start_server.sh`: Launches Uvicorn. The server binds to `0.0.0.0:8000`.

Data flow: Client -> FastAPI route (`openai_api.py`) -> prompt assembly -> `QwenLocalModel.generate_async()` -> vLLM -> text -> OpenAI-style JSON response.

## Conventions and patterns
- The API returns OpenAI-style `chat.completion` objects. Stick to the current fields (`id`, `object`, `created`, `model`, `choices[0].message.content`, `finish_reason`).
- Prompt format is simple role-tagged text:
  - `[System]: ...`, `[User]: ...`, `[Assistant]: ...`, ending with `[Assistant]:` to elicit a reply.
  - Default stop sequence includes `"\n[User]:"` to halt generation before the next user tag.
- Logging: Use `logging.getLogger(__name__)` and info-level logs for request metadata and messages. Avoid emojis; logs should be plain text.
- vLLM outputs can be empty in edge cases. `app/model.py` defensively returns `""` instead of raising when no outputs are present.

## Developer workflows
- Install deps: `pip install -r requirements.txt` (GPU + CUDA recommended for vLLM).
- Run server: `./start_server.sh` (exports `PYTHONPATH=.` and runs Uvicorn at `0.0.0.0:8000`).
- Try the client: `python3 example/client.py` (uses `AsyncOpenAI` with `base_url` pointing to this server).
- Change model: edit `QwenLocalModel.__init__(model_name=...)` in `app/model.py`. Keep `trust_remote_code=True` if using HF models that require it.
- Sampling: pass `temperature`, `max_tokens`, and `stop` via request; `top_p` is set server-side to 0.95 unless changed.

## Error handling expectations
- Do not crash on empty generations. `generate()` must guard list indexing and return `""` on failure, logging a warning. The HTTP layer should still return 200 with an empty `content` if generation failed non-fatally.
- If you add new routes or streaming, mirror OpenAI’s shapes (e.g., SSE for streaming) and keep compatibility.

## Cross-file contracts
- `openai_api.py` depends on `QwenLocalModel.generate_async(prompt, params)` returning a string. Preserve this contract.
- `SamplingParams` are created in the route with client-provided temperature/max_tokens/stop and passed into the model. The model should respect these if provided.

## Examples
- Request (client):
  - messages: `[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Tell me a joke."}]`
  - stop: `["\n[User]:"]`
- Response (shape):
  - `{"choices":[{"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}]}`

## What to avoid
- Changing response schema unexpectedly.
- Introducing blocking operations in the request path; use the thread pool or async patterns as shown.
- Leaking GPU memory: if you add long-lived objects, ensure lifecycle management.

## Useful files to read first
- `app/openai_api.py` for request/response shape and logging.
- `app/model.py` for vLLM usage and defensive output handling.
- `example/client.py` for how clients interact with this server.

If anything is unclear (e.g., model switching, streaming support, or additional endpoints), leave a short note in this file and open a small PR to align conventions.
