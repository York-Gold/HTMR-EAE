import torch
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .base_model import BaseLLMModel


class LocalLLMGenerator(BaseLLMModel):
    def __init__(
        self, model_path, temperature=0.01, max_new_tokens=8192, peft_model_path=None
    ):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if peft_model_path:
            self.model = PeftModel.from_pretrained(self.model, peft_model_path)
        self.model.eval()

    def infer(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,  # Use greedy decoding, the model always outputs the token with the highest probability
            do_sample=False,  # This must be specified, otherwise the model's response is very short
            max_new_tokens=self.max_new_tokens,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        # Output length statistics
        logging.info(f"Output Length: {len(output_ids)}")
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(
            "\n"
        ), len(output_ids)

    def generate(self, prompt):
        system = """
            You are a helpful AI assistant specialized in event argument extraction tasks. Please output according to the following format:
            <think> Your reasoning process </think> Your final extraction structure (JSON format)
        """
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.infer(message)
