from vllm import LLM, SamplingParams

class QwenLocalModel:
    def __init__(self, model_name="microsoft/Phi-4-mini-instruct"):
        self.llm = LLM(model=model_name, trust_remote_code=True)
        self.params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

    def generate(self, prompt: str, params = None) -> str:
        if params is None:
            params = self.params
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text
