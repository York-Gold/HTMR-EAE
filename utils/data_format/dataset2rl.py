"""
To convert ACE-E, ACE-Plus, or ERE datasets to RL format for reinforcement learning.
The conversion involves transforming the original dataset format into a structured prompt-completion format suitable for RL training.
Original

dataset example:
{
    "doc_id": "CNNHL_ENG_20030424_123502.25",
    "sent_id": "CNNHL_ENG_20030424_123502.25-2",
    "tokens": ["it", "happened", "at", "7", ":", "30", "this", "morning", "."],
    "pieces": ["it", "h", "app", "ened", "at", "7", ":", "30", "this", "morning", "."],
    "token_lens": [1, 3, 1, 1, 1, 1, 1, 1, 1],
    "sentence": "it happened at 7:30 this morning.",
    "entity_mentions": [],
    "relation_mentions": [],
    "event_mentions":
    [
        {
            "id": "CNNHL_ENG_20030424_123502.25-EV2-2",
            "event_type": "Life:Die",
            "trigger": {"text": "it", "start": 0, "end": 1},
            "arguments": []
        }
    ]
}

To convert to RL training format
{
    "data_source": "eae",
    "prompt": [
        {
            "role": "system",
            "content": "You are a helpful AI assistant specialized in event argument extraction tasks. Please output according to the following format: <think> Your reasoning process </think> Your final extraction structure (JSON format).",
        },
        {
            "role": "user",
            "content": "Extract event arguments from the following input: {input_json_str}"
        },
    ],
    "ability": "eae",
    "completion": "",
    "reward_model": {
        "style": "rule",
        "ground_truth": "[{role:..., entity:...}, ...]"
    },
    "extra_info": None
}
"""

import json

from runner.baseline_runner import BaselineRunner
from config import get_baseline_runner_config
from utils.utils import save_json_file


def convert_dataset_to_rl(args):
    config = get_baseline_runner_config(args)
    runner = BaselineRunner(config)
    openai_data = []
    for input_json_str, event_mention in runner.run():
        final_extractor_prompt = runner.extractor_prompt.replace(
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
        output_dict = {
            "data_source": "eae",
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant specialized in event argument extraction tasks. Please output according to the following format: <think> Your reasoning process </think> Your final extraction structure (JSON format).",
                },
                {"role": "user", "content": final_extractor_prompt},
            ],
            "ability": "eae",
            "completion": "",
            "reward_model": {"style": "rule", "ground_truth": answer_str},
            "extra_info": None,
        }
        openai_data.append(output_dict)
    save_json_file(config.result_path, openai_data)
    print(f"Converted {len(openai_data)} examples to {config.result_path}")
    return openai_data
