from vllm import LLM, SamplingParams

class QwenLocalModel:
    def __init__(self, model_name="Qwen/Qwen1.5-4B"):
        self.llm = LLM(model=model_name, trust_remote_code=True)
        self.params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.params)
        return outputs[0].outputs[0].text
