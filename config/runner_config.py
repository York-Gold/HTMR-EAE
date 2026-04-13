from model import LocalLLMGenerator, LocalVLLMGenerator, BaseLLMModel
import logging
from dataclasses import dataclass


@dataclass
class BaselineRunnerConfig:
    entity_dict_path: str
    template_path: str
    prompt_path: str
    result_path: str
    llm: BaseLLMModel
    test_data_path: str
    is_ace_plus_data: bool
    use_vllm: bool = False


def get_baseline_runner_config(args):
    if not hasattr(args, "use_vllm"):
        args.use_vllm = False
    if not hasattr(args, "model_path"):
        args.model_path = None
    if args.use_vllm and args.model_path:
        llm_generator = LocalVLLMGenerator(args.model_path)
    elif args.model_path:
        llm_generator = LocalLLMGenerator(args.model_path)
    else:
        llm_generator = None
    if args.dataset_type == 0:
        logging.info("Creating BaselineRunnerConfig for ACE dataset.")
        # ACE Dataset
        return BaselineRunnerConfig(
            entity_dict_path="datasets/ACE-E/ace_entity_map.json",
            template_path="datasets/ACE-E/ace_template.json",
            test_data_path=(
                "datasets/ACE-E/test.oneie.jsonl"
                if not hasattr(args, "test_data_path") or not args.test_data_path
                else args.test_data_path
            ),
            prompt_path="prompts/extractor-ace-e.txt",
            result_path=args.result_path,
            llm=llm_generator,
            is_ace_plus_data=False,
            use_vllm=args.use_vllm,
        )

    elif args.dataset_type == 1:
        logging.info("Creating BaselineRunnerConfig for ACE Plus dataset.")
        # ACE Plus Dataset
        return BaselineRunnerConfig(
            entity_dict_path="datasets/ACE-E-Plus/ace_entity_map.json",
            template_path="datasets/ACE-E-Plus/ace_template.json",
            test_data_path=(
                "datasets/ACE-E-Plus/test.oneie.jsonl"
                if not hasattr(args, "test_data_path") or not args.test_data_path
                else args.test_data_path
            ),
            prompt_path="prompts/extractor-ace-ep.txt",
            result_path=args.result_path,
            llm=llm_generator,
            is_ace_plus_data=True,
            use_vllm=args.use_vllm,
        )
    elif args.dataset_type == 2:
        logging.info("Creating BaselineRunnerConfig for ERE dataset.")
        # ERE Dataset
        return BaselineRunnerConfig(
            entity_dict_path="datasets/ERE/ere_entity_map.json",
            template_path="datasets/ERE/ere_template.json",
            test_data_path=(
                "datasets/ERE/test.json"
                if not hasattr(args, "test_data_path") or not args.test_data_path
                else args.test_data_path
            ),
            prompt_path="prompts/extractor-ere.txt",
            result_path=args.result_path,
            llm=llm_generator,
            is_ace_plus_data=False,
            use_vllm=args.use_vllm,
        )
    else:
        raise ValueError("Invalid dataset type. Please choose from [0, 1, 2].")
