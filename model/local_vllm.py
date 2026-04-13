from vllm import LLM, SamplingParams
from tqdm import tqdm

from .base_model import BaseLLMModel


class LocalVLLMGenerator(BaseLLMModel):
    def __init__(self, model_path, temperature=0.01, max_new_tokens=8192):
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            seed=42,
            n=1,
            best_of=1,
            repetition_penalty=1,
            max_tokens=max_new_tokens,
        )
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,  # Whether to use multiple GPUs
        )

    # Utilize vllm's multi-concurrency feature to process multiple inputs at once
    def generate(self, texts):
        system = """
            You are a helpful AI assistant specialized in event argument extraction tasks. Please output according to the following format:
            <think> Your reasoning process </think> Your final extraction structure (JSON format)
        """
        messages = []
        for text in texts:
            message = [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ]
            messages.append(message)
        return self.infer(messages)

    def infer(self, messages):
        if len(messages) == 1:
            return self._infer_single(messages[0])
        else:
            return self._infer_batch(messages)

    def _infer_single(self, messages):
        prompt = self.model.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        response = self.model.generate(
            [prompt],
            sampling_params=self.sampling_params,
        )
        return response[0].outputs[0].text.strip("\n"), len(
            response[0].outputs[0].token_ids
        )

    # When batch_size is None, it means using all inputs for inference
    # When batch_size is 1, it acts like calling _infer_single
    def _infer_batch(self, messages_list, batch_size=None):
        if batch_size is None:
            batch_size = len(messages_list)
        results = []
        results_length = []
        for i in tqdm(range(0, len(messages_list), batch_size), desc="Model Inference"):
            batch_messages = messages_list[i : i + batch_size]
            prompts = [
                self.model.get_tokenizer().apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                for messages in batch_messages
            ]
            responses = self.model.generate(
                prompts,
                sampling_params=self.sampling_params,
            )
            for response in responses:
                results.append(
                    response.outputs[0].text.strip("\n"),
                )
                results_length.append(
                    len(
                        response.outputs[0].token_ids,
                    )
                )
        return results, results_length
