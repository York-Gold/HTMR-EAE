"""
Base class for LLM models.
"""

class BaseLLMModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("The subclass must implement this method")
    
    def infer(self, messages: list) -> str:
        raise NotImplementedError("The subclass must implement this method")