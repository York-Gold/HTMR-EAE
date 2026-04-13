"""
Baseline Runner for Event Argument Extraction Task
"""

import json
import traceback
import logging

from utils import read_json, save_json_file, read_json_skip_no_event, read_txt_file
from config import BaselineRunnerConfig
from tqdm import tqdm


entity_mention_map = {
    "FAC": "Facility",
    "GPE": "Geo-Political Entity",
    "PER": "Person",
    "ORG": "Organization",
    "VEH": "Vehicle",
    "WEA": "Weapon",
    "LOC": "Location",
}


class BaselineRunner:
    def __init__(
        self,
        config: BaselineRunnerConfig,
    ):
        self.entity_dict = read_json(config.entity_dict_path)
        self.template = read_json(config.template_path)
        self.is_ace_plus_data = config.is_ace_plus_data
        # Need to skip data without event mentions
        self.data_set_json_list = read_json_skip_no_event(config.test_data_path)
        self.extractor_prompt = read_txt_file(config.prompt_path)
        self.result_path = config.result_path
        self.llm = config.llm
        self.use_vllm = config.use_vllm
    
    def run(self):
        offset = 0
        self.result_json_list = []
        for index, data_set_json in tqdm(enumerate(self.data_set_json_list), total=len(self.data_set_json_list)):
            try:
                if index < offset:
                    continue
                # Used to distinguish between ace dataset and ere dataset, ere dataset does not have sentence field, need to concatenate tokens field
                self.sentence = data_set_json.get("sentence") or " ".join(data_set_json["tokens"])
                self.entity_mentions = data_set_json["entity_mentions"]
                self.event_mentions = data_set_json["event_mentions"]

                # Simplify entity mention information, only keep text, entity type, and entity subtype
                self.easy_entity_mentions = self.simplify_entity_mentions()
                for event_mention in self.event_mentions:
                    input_json_str = self.preprocess_input_json(event_mention)
                    yield input_json_str, event_mention
            except Exception:
                print(f"Error Index: {index}\n\nDetail: {traceback.format_exc()}")
                ret = {}
                ret["sentence"] = data_set_json.get("sentence") or " ".join(data_set_json["tokens"])
                ret["error"] = (
                    f"Error Index: {index}\n\nDetails: {traceback.format_exc()}"
                )
                self.result_json_list.append(ret)

                save_json_file(self.result_path, self.result_json_list)

    # Simplify entity mentions information
    def simplify_entity_mentions(self):
        easy_entity_mentions = []
        for entity_mention in self.entity_mentions:
            easy_entity_mentions.append(
                {
                    "entity": entity_mention["text"],
                    "entity_type": entity_mention_map[entity_mention["entity_type"]],
                    "entity_subtype": entity_mention["entity_subtype"] if self.is_ace_plus_data else None,
                }
            )
        # Remove fields with None values
        easy_entity_mentions = [
            self.process_dict(item) for item in easy_entity_mentions
        ]
        return easy_entity_mentions

    # Preprocess input json for the model
    def preprocess_input_json(self, event_mention):
        event_type = event_mention["event_type"].replace(":", ".")
        input_json = {
            "sentence": self.sentence,
            "entity_mentions": self.easy_entity_mentions,
            "trigger": { 
                "word": event_mention["trigger"]["text"],
                "event_type": event_type,
            },
            "argument_roles": self.entity_dict["event_role_entity_map"][event_type],
            "template": self.template[event_type],
        }
        input_json_str = json.dumps(input_json, ensure_ascii=False)
        
        return input_json_str

    # Remove fields with None values
    def process_dict(self, dict_data):
        keys_to_remove = [key for key, value in dict_data.items() if value is None]
        for key in keys_to_remove:
            del dict_data[key]
        return dict_data

    def save_final_result(self, result_json_list, event_mention, extractor_output, output_length):
        result_json_list.append(
            {
                "sentence": self.sentence,
                "entity_mentions": self.easy_entity_mentions,
                "trigger": {
                    "word": event_mention["trigger"]["text"],
                    "event_type": event_mention["event_type"].replace(":", "."),
                },
                "extractor_output": extractor_output,
                "output_length": output_length,
                "answer": event_mention,
            }
        )
        save_json_file(self.result_path, result_json_list)

    def _predict_single(self):
        for input_json_str, event_mention in tqdm(self.run(), desc="Model Inference", total=len(self.data_set_json_list)):
            final_extractor_prompt = self.extractor_prompt.replace(
                "[Insert Input Here]", input_json_str
            )
            extractor_output, output_length = self.llm.generate(final_extractor_prompt)
            self.save_final_result(
                self.result_json_list, event_mention, extractor_output, output_length
            )

    def _predict_batch(self):
        final_extractor_prompt_list = []
        event_mention_list = []
        for input_json_str, event_mention in self.run():
            final_extractor_prompt = self.extractor_prompt.replace(
                "[Insert Input Here]", input_json_str
            )
            final_extractor_prompt_list.append(final_extractor_prompt)
            event_mention_list.append(event_mention)
        logging.info(f"Total Prompts will to be Processed: {len(final_extractor_prompt_list)}")
        extractor_output_list, output_length_list = self.llm.generate(final_extractor_prompt_list)
        for i in range(len(final_extractor_prompt_list)):
            event_mention = event_mention_list[i]
            extractor_output = extractor_output_list[i]
            output_length = output_length_list[i]
            self.save_final_result(
                self.result_json_list, event_mention, extractor_output, output_length
            )

    def predict(self):
        if self.use_vllm:
            self._predict_batch()
        else:
            self._predict_single()