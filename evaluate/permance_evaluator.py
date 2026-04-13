import json
import ast
import logging


class PerformanceEvaluator:
    """
    Args:
        gt_results: Real parameter list, such as [[{"text": "Bush", "role": "Entity"},...],...]
        pred_results: List of predicted parameters, such as [[{"role": "Entity", "entity": "Bush"},...],...]
        sentences: Original sentence list for each sample
        The three have the same length
    """

    def __init__(self, result_path):
        self.result_dict = json.load(open(result_path, "r"))
        self.sentences = []
        self.gt_results = []
        self.pred_results = []
        self.output_length = []
        self.get_results()

    def evaluate(self):
        # Define the indicators for argument classification, representing the true quantity, predicted quantity, and correct quantity respectively
        arg_c_gt_num, arg_c_pred_num, arg_c_correct_num = 0, 0, 0
        for i in range(len(self.gt_results)):
            pred_i = self.pred_results[i]
            gt_i = self.gt_results[i]

            arg_c_gt, arg_c_pred, arg_c_corr = self.evaluate_arg_c(pred_i, gt_i)
            arg_c_gt_num += arg_c_gt
            arg_c_pred_num += arg_c_pred
            arg_c_correct_num += arg_c_corr

        rpf_arg_c = self.cal_rpf(arg_c_gt_num, arg_c_pred_num, arg_c_correct_num)
        logging.info(f"Final Arg-C Score: {rpf_arg_c}")
        logging.info(
            f"Average Output Length: {sum(self.output_length)/len(self.output_length)}"
        )
        return rpf_arg_c

    # Argument Classification, Arg-C
    def evaluate_arg_c(self, pred_i, gt_i):
        gt_num, pred_num, correct_num = 0, 0, 0
        gt_num += len(gt_i)
        pred_num += len(pred_i)

        for gt_i_json in gt_i:
            role = gt_i_json["role"]
            text = gt_i_json["text"]
            for pred_i_json in pred_i:
                if not isinstance(pred_i_json, dict):
                    continue
                if pred_i_json["role"] == role and pred_i_json["entity"] == text:
                    correct_num += 1
                    break
        return gt_num, pred_num, correct_num

    # Calculate P/R/F1 metrics based on the results
    def cal_rpf(self, gt_num, pred_num, correct_num):
        precision = correct_num / pred_num if pred_num > 0 else 0.0
        recall = correct_num / gt_num if gt_num > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 1e-4
            else 0.0
        )
        return {
            "P": precision,
            "R": recall,
            "F1": f1,
            "gt_num": gt_num,
            "pred_num": pred_num,
            "correct_num": correct_num,
        }

    # Get ground truth and predicted results
    def get_results(self):
        for item in self.result_dict:
            self.pred_results.append(self.extract_json(item["extractor_output"]))
            self.output_length.append(item["output_length"])
            self.gt_results.append(item["answer"]["arguments"])
            self.sentences.append(item["sentence"])

    # Extract JSON predicted results from model output
    def extract_json(self, model_output):
        try:
            result = model_output.split("</think>")[1]
            left = result.rfind("[")
            right = result.rfind("]")
            json_str = result[left : right + 1]

            if json_str == "":
                return []
            return ast.literal_eval(json_str)
        except:
            # logging.error(
            #     f"Failed to parse JSON from model output: {model_output[-100:]}"
            # )
            return []
