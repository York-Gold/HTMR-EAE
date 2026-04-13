# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import ast
import json

def extract_output(json_data):
    result = json_data.split("</think>")[1]
    left = result.rfind("[")
    right = result.rfind("]")
    json_str = result[left : right + 1]

    if json_str == "":
        return ""
    return ast.literal_eval(json_str)

def compute_score(solution_str, ground_truth, data_source=None, extra_info=None):
    try:
        if "</think>" not in solution_str or "<think>" not in solution_str:
            return 0.0
        pred_dict = extract_output(solution_str)

        gt_dict = json.loads(ground_truth)

        all_gt_num = len(gt_dict)
        all_pred_num = len(pred_dict)
        all_bigger_num = max(all_gt_num, all_pred_num)
        
        if all_gt_num == 0:
            if all_pred_num == 0:
                return 1.0
            else:
                return max(0.4-0.1*all_bigger_num, 0.1) 
        else:
            correct_num = 0
            for gt_i in gt_dict:
                for pred_i in pred_dict:
                    if gt_i["role"] == pred_i.get("role", "") and gt_i["entity"] == pred_i.get("entity", ""):
                        correct_num += 1
                        break
            return max(correct_num / all_bigger_num, 0.4-0.1*all_bigger_num, 0.1)
    except:
        # print(f"Failed to parse JSON from model output: {solution_str}")
        # import traceback; traceback.print_exc()  
        return 0.0