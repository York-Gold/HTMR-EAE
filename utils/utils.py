import json


def read_json_skip_no_event(file_path):
    """
    Read JSON or JSONL file and skip entries without event mentions
    """
    data = []
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            for item in all_data:
                if item["event_mentions"] != []:
                    data.append(item)
        return data
    elif file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if json.loads(line)["event_mentions"] != []:
                    data.append(json.loads(line))
        return data


def read_json(file_path):
    """
    Read JSON or JSONL file and return the content as a list or dictionary

    Args:
        file_path: File path

    Returns:
        If it is a .jsonl file, return a list (each line is a dictionary); if it is a .json file, return a dictionary or list
    """
    if file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line))
        return data
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_json_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
