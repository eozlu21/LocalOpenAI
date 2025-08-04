import asyncio
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams

class QwenLocalModel:
    def __init__(self, model_name="microsoft/Phi-4-mini-instruct"):
        self.llm = LLM(model=model_name, trust_remote_code=True)
        self.params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        # Thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=2)

    def generate(self, prompt: str, params = None) -> str:
        if params is None:
            params = self.params
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text
    
    async def generate_async(self, prompt: str, params = None) -> str:
        """Async wrapper for generate method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate, prompt, params)
    
    def __del__(self):
        """Cleanup thread pool on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
