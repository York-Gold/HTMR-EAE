import logging

from dataclasses import dataclass


@dataclass
class BaseLineGenerationConfig:
    result_path: str
    train_dataset_path: str
    log_path: str


def get_baseline_generation_config(args):
    if args.dataset_type == 0:
        logging.info("Creating BaseLineGenerationConfig for ACE dataset.")
        # ACE Dataset
        return BaseLineGenerationConfig(
            result_path=args.result_path,
            train_dataset_path="datasets/ACE-E/train.oneie.jsonl",
            log_path=args.log_path
        )
    elif args.dataset_type == 1:
        logging.info("Creating BaseLineGenerationConfig for ACE Plus dataset.")
        # ACE Plus Dataset
        return BaseLineGenerationConfig(
            result_path=args.result_path,
            train_dataset_path="datasets/ACE-E-Plus/train.oneie.jsonl",
            log_path=args.log_path
        )
    elif args.dataset_type == 2:
        logging.info("Creating BaseLineGenerationConfig for ERE dataset.")
        # ERE Dataset
        return BaseLineGenerationConfig(
            result_path=args.result_path,
            train_dataset_path="datasets/ERE/train.json",
            log_path=args.log_path
        )
    else:
        raise ValueError("Invalid dataset_type. Choose from [0, 1, 2].")
