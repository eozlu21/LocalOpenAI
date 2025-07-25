from fastapi import FastAPI
from app.openai_api import router as openai_router

app = FastAPI(
    title="Local OpenAI-Compatible API",
    description="Runs a Qwen model and mimics OpenAI's ChatCompletion interface.",
    version="1.0"
)

# Mount the OpenAI-compatible route
app.include_router(openai_router)
