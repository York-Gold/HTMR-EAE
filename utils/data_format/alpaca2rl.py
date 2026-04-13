"""
To convert Alpaca formatted data to RL format for reinforcement learning.
The converted dataset will include a new field "ground_truth".
The Alpaca format:
    {
        "instruction": "...",
        "input": "...",  # can be empty
        "output": "..."
    }
The RL format:
    {
        "data_source": "eae",
        "prompt": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant specialized in event argument extraction tasks. Please
                output according to the following format: <think> Your reasoning process </think> Your final extraction structure (JSON format).",
            },
            {
                "role": "user",
                "content": "..."
            },
        ],
        "ability": "eae",
        "completion": "...",
        "reward_model": {
            "style": "rule",
            "ground_truth": "..."
        },
        "extra_info": None  
    }
"""

import json
import ast
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.utils import save_json_file


def convert_alpaca_to_rl(args):
    with open(args.input_file, "r", encoding="utf-8") as f:
        alpaca_data = json.load(f)

    openai_data = []

    for i, entry in enumerate(alpaca_data):
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        output_text = entry.get("output", "")

        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        answer_str = json.dumps(
            extract_ground_truth(output_text), indent=0, ensure_ascii=False
        ).replace("\n", "")

        openai_entry = {
            "data_source": "eae",
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant specialized in event argument extraction tasks. Please output according to the following format: <think> Your reasoning process </think> Your final extraction structure (JSON format).",
                },
                {"role": "user", "content": prompt},
            ],
            "ability": "eae",
            "completion": f"{output_text}",
            "reward_model": {"style": "rule", "ground_truth": answer_str},
            "extra_info": None,
        }
        openai_data.append(openai_entry)
    save_json_file(args.output_file, openai_data)
    print(f"Converted {len(openai_data)} examples to {args.output_file}")
    return openai_data


def extract_ground_truth(json_data):
    try:
        result = json_data.split("</think>")[1]
        left = result.rfind("[")
        right = result.rfind("]")
        json_str = result[left : right + 1]

        if json_str == "":
            return []
        return ast.literal_eval(json_str)
    except:
        print(f"Failed to parse JSON from model output: {json_data[-100:]}")
        return []


# Convert the dataset from JSON format to Parquet format
def json_to_parquet(json_data, parquet_file):
    df = pd.read_json(json.dumps(json_data))
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_file)
