import ast
import pdb


def extract_json_old(string):
    try:
        result = string.split("</think>")[1]
        left_1 = result.find("{")
        left_2 = result.rfind("[")
        if left_1 != -1 and left_1 < left_2:
            right = result.rfind("}")
            json_str = result[left_1 : right + 1]
        else:
            right = result.rfind("]")
            json_str = result[left_2 : right + 1]

        if json_str == "":
            return {}
        return ast.literal_eval(json_str)
    except:
        return {}


def extract_json(string):
    try:
        result = string.split("</think>")[1]
        left = result.rfind("[")
        right = result.rfind("]")
        json_str = result[left : right + 1]

        if json_str == "":
            return []
        return ast.literal_eval(json_str)
    except:
        return []