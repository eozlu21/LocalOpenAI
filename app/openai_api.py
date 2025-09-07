from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Literal
import asyncio
import time
import logging

from vllm import SamplingParams
from app.model import QwenLocalModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    stop: Optional[List[str]] = None

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Log incoming request
    logger.info("Incoming chat completion request:")
    logger.info(f"   Model: {req.model}")
    logger.info(f"   Temperature: {req.temperature}")
    logger.info(f"   Max tokens: {req.max_tokens}")
    logger.info(f"   Stop: {req.stop}")
    
    # Log messages
    logger.info("Messages:")
    for i, m in enumerate(req.messages):
        logger.info(f"   [{i+1}] {m.role.upper()}: {m.content}")
    
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
    
    logger.info("Generating response...")
    
    # Run the model generation asynchronously
    result = await model.generate_async(prompt, params)
    
    logger.info(f"Response generated: {result.strip()[:100]}{'...' if len(result.strip()) > 100 else ''}")

    return {
        "id": "chatcmpl-local-qwen",
        "object": "chat.completion",
        "created": int(time.time()),
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