from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Literal

from vllm import SamplingParams
from app.model import QwenLocalModel

router = APIRouter()
model = QwenLocalModel()

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = 512
    stop: Optional[List[str]] = None  # NEW

@router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    prompt = ""
    for m in req.messages:
        if m.role == "system":
            prompt += f"[System]: {m.content}\n"
        elif m.role == "user":
            prompt += f"[User]: {m.content}\n"
        elif m.role == "assistant":
            prompt += f"[Assistant]: {m.content}\n"

    prompt += "[Assistant]:"

    params = SamplingParams(
        temperature=req.temperature,
        top_p=0.95,
        max_tokens=req.max_tokens,
        stop=req.stop or ["[User]:"]
    )
    result = model.generate(prompt, params)

    return {
        "id": "chatcmpl-local-qwen",
        "object": "chat.completion",
        "created": 1234567890,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.strip()
                },
                "finish_reason": "stop"
            }
        ]
    }
