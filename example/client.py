from openai import OpenAI

# === Configuration ===
API_BASE = "http://ai21.kuvalar.ku.edu.tr:8000/v1"
MODEL    = "microsoft/Phi-4-mini-instruct"

# === Instantiate client ===
client = OpenAI(
    api_key="not-needed",    # ignored by local server
    base_url=API_BASE,       # points to your FastAPI endpoint
)

# === Send request ===
response = client.chat.completions.create(
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
