# 📃 LocalOpenAI — OpenAI-Compatible Local API Server (Qwen-based)

This project serves a Qwen model locally using `FastAPI` and exposes a REST API compatible with the OpenAI `ChatCompletion` interface (`/v1/chat/completions`). You can use it with `curl`, Postman, or the official OpenAI Python client.

---

## 🚀 Features

* Uses [Qwen1.5-4B](https://huggingface.co/Qwen/Qwen1.5-4B) or any Hugging Face-compatible model.
* REST API mimics OpenAI's `chat/completions` endpoint.
* Can be used by multiple users or clients (e.g. via `openai.ChatCompletion.create()`).
* FastAPI + vLLM backend.
* Designed for HPC clusters or local development environments.

---

## 📁 Directory Structure

```
LocalOpenAI/
├── app/
│   ├── main.py           # FastAPI server
│   ├── openai_api.py     # OpenAI-compatible endpoint
│   └── model.py          # Qwen model interface (via vLLM)
├── requirements.txt
├── start_server.sh       # Launch server with logs
└── README.md             # This file
```

---

## 🔧 Step 1: Set Up Environment (Micromamba)

> Micromamba is preferred for isolated, fast Python environments.

```bash
# Optional: Add micromamba to your PATH if not already
export PATH=~/micromamba/bin:$PATH

# Create environment
micromamba create -n localopenai python=3.10 -y
micromamba activate localopenai

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Step 2: Start the Model Server

You must launch this from a compute node (e.g. `ai21`):

```bash
./start_server.sh
```

This runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You should see:

```
Uvicorn running on http://0.0.0.0:8000
```

---

## 📬 Step 3: Send Requests (via `curl` or OpenAI client)

### ✅ Using `curl` (from another node like login02):

```bash
curl http://ai21.kuvalar.ku.edu.tr:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen1.5-4B", "messages": [{"role": "user", "content": "Tell me a joke."}]}'
```

---

### ✅ Using OpenAI Python Client:

```python
import openai

openai.api_key = "not-needed"
openai.base_url = "http://ai21.kuvalar.ku.edu.tr:8000/v1"

response = openai.ChatCompletion.create(
    model="Qwen1.5-4B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the World Cup in 2022?"}
    ]
)

print(response.choices[0].message["content"])
```

---

## 🔍 Troubleshooting

* **`curl` hangs or shows `>` prompt** → Make sure your JSON is valid and single-quoted properly.
* **`Connection refused`** → Ensure the server is running on the correct node with `--host 0.0.0.0`.
* **No response or empty message** → Check logs in the terminal running `start_server.sh`.

---

## ✉️ Contributing

Pull requests and improvements are welcome. For issues or feature requests, please open a GitHub issue.

---

## 🙏 Acknowledgements

* [Qwen Model](https://huggingface.co/Qwen)
* [vLLM Project](https://github.com/vllm-project/vllm)
* [FastAPI](https://fastapi.tiangolo.com)
* OpenAI Python SDK
