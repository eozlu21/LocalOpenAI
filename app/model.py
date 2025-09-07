import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from vllm import LLM, SamplingParams


class QwenLocalModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B-Instruct") -> None:  # "microsoft/Phi-4-mini-instruct"
        # Allow environment override (set by CLI wrapper script)
        env_model = os.getenv("LOCAL_OPENAI_MODEL")
        if env_model:
            model_name = env_model

        tp_env = os.getenv("LOCAL_OPENAI_TP")
        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.80,
        }
        if tp_env:
            try:
                tp = int(tp_env)
                if tp > 0:
                    llm_kwargs["tensor_parallel_size"] = tp
            except ValueError:
                pass  # ignore invalid value

        self.llm = LLM(**llm_kwargs)
        self.params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        # Thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str, params: Optional[SamplingParams] = None) -> str:
        if params is None:
            params = self.params

        try:
            request_outputs = self.llm.generate([prompt], params)
        except Exception as e:  # pragma: no cover - defensive guard
            # Log and return empty text to keep API stable
            self.logger.exception("vLLM generation failed: %s", e)
            return ""

        # vLLM returns a list of RequestOutput objects. In rare cases (e.g.,
        # immediate stop sequence match or internal cancellation), the nested
        # outputs list can be empty. Guard against that to avoid IndexError.
        if not request_outputs:
            self.logger.warning("vLLM returned no request outputs; returning empty text.")
            return ""

        ro = request_outputs[0]
        outputs = getattr(ro, "outputs", None) or []
        if not outputs:
            self.logger.warning("vLLM returned an empty outputs list; returning empty text.")
            return ""

        # Choose the first candidate with text if available
        for cand in outputs:
            text = getattr(cand, "text", None)
            if isinstance(text, str):
                return text

        self.logger.warning("vLLM outputs contained no text attributes; returning empty text.")
        return ""

    async def generate_async(self, prompt: str, params: Optional[SamplingParams] = None) -> str:
        """Async wrapper for generate method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate, prompt, params)

    def __del__(self) -> None:
        """Cleanup thread pool on destruction"""
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=True)
        except Exception:
            # best-effort cleanup
            pass
