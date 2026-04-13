from openai import OpenAI
import os


def get_answer_by_qwen3max(prompt):
    client = OpenAI(
        api_key=os.getenv("QWEN3MAX_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    ret = {
        "content": completion.choices[0].message.content,
    }
    return ret