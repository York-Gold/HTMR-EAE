"""
Baseline Generation Module, for generating baseline data using CoT prompts and a language model API.
This module extends the BaselineRunner to implement generation functionality.
"""

import json
import logging
import os
import random

from runner.baseline_runner import BaselineRunner
from utils.api_funcs import get_answer_by_qwen3max
from utils import save_json_file, read_json_skip_no_event, read_txt_file
from config import BaseLineGenerationConfig, BaselineRunnerConfig

COT_PROMPT_DIR = "prompts/multi/"
random.seed(42)


class BaselineGeneration(BaselineRunner):
    def __init__(
        self,
        run_config: BaselineRunnerConfig,
        gen_config: BaseLineGenerationConfig,
    ):
        super().__init__(run_config)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filemode="a",
            filename=gen_config.log_path,
            force=True,
        )
        # Modify two parent class attributes
        self.data_set_json_list = read_json_skip_no_event(gen_config.train_dataset_path)
        self.result_path = gen_config.result_path

    def generate(self):
        result_json_list = []
        total = len(self.data_set_json_list)
        logging.info(f"Total Data: {total}")
        cnt = 0
        # Prevent interruption
        offset = 0
        for input_json_str, event_mention in self.run():
            # Get a random cot_extractor_prompt, place it before offset to ensure cot_prompt remains unchanged after resuming from interruption
            cot_extractor_prompt = self.get_random_cot_prompt()
            if cnt < offset:
                logging.info(
                    f"[Sample {cnt+1}/{total}] skipped successfully."
                )
                cnt += 1
                continue
            
            final_extractor_prompt = self.extractor_prompt.replace(
                "[Insert Input Here]", input_json_str
            )
            answer_json = []
            arguments = event_mention["arguments"]
            for argument in arguments:
                answer_json.append(
                    {
                        "role": argument["role"],
                        "entity": argument["text"],
                    }
                )
            answer_str = json.dumps(answer_json, indent=0, ensure_ascii=False).replace(
                "\n", ""
            )
            coter_prompt = cot_extractor_prompt.replace(
                "[Insert Input Here]", input_json_str
            ).replace("[Insert Output Here]", answer_str)
            logging.info(f"[Sample {cnt+1}/{total}] Begin API call. ")
            try:
                cot_ans = get_answer_by_qwen3max(coter_prompt)["content"]
                logging.info(
                    f"[Sample {cnt+1}/{total}] API call finished successfully."
                )
            except Exception as e:
                logging.error(f"[Sample {cnt+1}/{total}] API call failed: {e}")
                cot_ans = "[API ERROR]"

            output_str = f"<think> {cot_ans} Therefore, the output should be {answer_str}.\n</think>\n\n```json\n{answer_str}\n```"

            result_json_list.append(
                {
                    "instruction": final_extractor_prompt,
                    "input": "",
                    "output": output_str,
                    "system": self.get_system_prompt(),
                }
            )
            save_json_file(self.result_path, result_json_list)
            logging.info(f"[Sample {cnt+1}/{total}] Result saved.\n")
            cnt += 1

    def get_system_prompt(self):
        return """You are a helpful AI assistant specialized in event argument extraction tasks. Please output according to the following format: <think> Your reasoning process </think> Your final extraction structure (JSON format)"""

    def get_random_cot_prompt(self):
        prompt_files = [
            os.path.join(COT_PROMPT_DIR, f)
            for f in os.listdir(COT_PROMPT_DIR)
            if os.path.isfile(os.path.join(COT_PROMPT_DIR, f))
        ]
        selected_file = random.choice(prompt_files)
        prompt = read_txt_file(selected_file)
        logging.info(f"Selected CoT prompt file: {selected_file}")
        return prompt
